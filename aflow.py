from sqlmodel import Field, Relationship, SQLModel, create_engine, Session as S, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
from PIL import Image
from io import BytesIO

from anode import custom
import base64
import argparse
import sys
import random
import os
from util import *
from tqdm import tqdm
import time
from db import *
import importlib.util
from florence import G_Dino
from typing import List, Dict, Any

optimize = None
def set_optimize(new_optimize):
    global optimize
    optimize = new_optimize

def get_correct_incorrect(runs):
    correct = []
    incorrect = []
    for run in runs:
        if run.score > 0.5:
            correct.append(run)
        else:
            incorrect.append(run)
    return correct, incorrect


async def main(args):
    graph = get_graph_from_a_file(args.seed_file)
    await graph.run(get_random_or_high_variance_task())

    strongest = None
    wins = 0
    best_score = -10000
    best_for = 0

    with S(es) as session:
        session.expire_on_commit = False
        all_graphs = session.exec(select(Graph)).all()
    await asyncio.gather(*[graph.run(get_random_or_high_variance_task()) for graph in all_graphs])
    import time
    time.sleep(4)
    for _ in tqdm(range(100)):
        if wins >= 5 or best_for >= 5:
            break

        with S(es) as session:
            session.expire_on_commit = False
            assert set(session.exec(
                select(Run.graph_id).group_by(Run.graph_id)
            ).all()) == set(
                session.exec(
                select(Graph.id)
            ).all())

        graphs = get_strongest_graph(3)
        tasks = get_random_or_high_variance_task(args.experiment_strategy)

        await asyncio.gather(*[graph.run(task) for graph in graphs for task in tasks])

        graph = get_strongest_graph()
        with S(es) as session:
            session.expire_on_commit = False
            graph = session.merge(graph)
            correct, incorrect = get_correct_incorrect(graph.runs)
            print(f"Current runs: {len(graph.runs)}, Correct: {len(correct)}, Incorrect: {len(incorrect)}")
        tot = 0
        while min(len(correct), len(incorrect)) < 5:
            tot += 1
            tasks = get_random_or_high_variance_task(args.experiment_strategy)
            print(f"Running additional tasks: {[task['id'] for task in tasks]}")
            await asyncio.gather(*[graph.run(task) for task in tasks])
            import time
            time.sleep(1)
            with S(es) as session:
                session.expire_on_commit = False
                graph = session.merge(graph)
                correct, incorrect = get_correct_incorrect(graph.runs)
                print(f"Current runs: {len(graph.runs)}, Correct: {len(correct)}, Incorrect: {len(incorrect)}")


        if graph == strongest:
            wins += 1
        else:
            strongest = graph,
            wins = 0

        if graph.score >= best_score:
            best_score = graph.score
            best_for = 0
        else:
            best_for += 1

        old_graph = graph
        graph = optimize(old_graph)

        for _ in range(3):
            try:
                await graph.run(get_random_or_high_variance_task())
                put(graph)
                await graph.run(get_random_or_high_variance_task(5))
                break
            except Exception as e:
                graph = debug(old_graph, graph, e)


def test_run(args):
    graph = get_graph_from_a_file(args.seed_file)
    task = get_random_task()
    print(asyncio.run(graph.run(task)))


def test_optimize(args):
    graph = get_graph_from_a_file(args.seed_file)
    graph = asyncio.run(optimize(graph))

def g_dino_detect(image, objects, box_threshold=0.3, text_threshold=0.25):
    from PIL import Image
    from florence import G_Dino

    if isinstance(image, str):
        image = Image.open(image)

    g_dino = G_Dino()
    detections = g_dino.detect(image, objects, box_threshold=box_threshold, text_threshold=text_threshold)
    return detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="task folder",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        help="initial graph python file",
    )
    parser.add_argument(
        "--data-factory",
        type=str,
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
    )
    parser.add_argument(
        "--db-name",
        type=str,
    )
    parser.add_argument(
        "--inference-model",
        type=str,
        default="plus",
    )
    parser.add_argument(
        "--optimization-model",
        type=str,
        default="max",
    )
    parser.add_argument(
        "--experiment-strategy",
        type=int,
        default=5,
        help="how many tasks to run per evaluation",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    model_map = {
        'plus': 'qwen-vl-plus-latest',
        'max': 'qwen-vl-max-latest',
        'claude': 'claude-3-7-sonnet-20250219',
    }
    if args.inference_model in model_map:
        args.inference_model = model_map[args.inference_model]
    else:
        # If the model name is not in the map, ensure it's a valid string
        if not args.inference_model or not isinstance(args.inference_model, str):
            args.inference_model = 'qwen-vl-plus-latest'  # Default to a safe value

    if args.optimization_model in model_map:
        args.optimization_model = model_map[args.optimization_model]
    else:
        # If the model name is not in the map, ensure it's a valid string
        if not args.optimization_model or not isinstance(args.optimization_model, str):
            args.optimization_model = 'qwen-vl-max-latest'  # Default to a safe value

    if not args.seed_file:
        args.seed_file = f"{args.folder}/seed.py"
    if not args.data_factory:
        args.data_factory = f"{args.folder}/data.py"
    if not args.prompt_file:
        args.prompt_file = f"{args.folder}/prompts.py"
    if not args.db_name:
        args.db_name = f"{args.folder}/db.sqlite"

    data_file_path = os.path.abspath(args.data_factory)

    data_spec = importlib.util.spec_from_file_location("data_module", data_file_path)
    data_module = importlib.util.module_from_spec(data_spec)
    data_spec.loader.exec_module(data_module)
    new_get_all_task_ids = getattr(data_module, "get_all_task_ids", None)
    new_get_task_by_id = getattr(data_module, "get_task_by_id", None)

    assert callable(new_get_all_task_ids) and callable(new_get_task_by_id)
    set_data(new_get_all_task_ids, new_get_task_by_id)

    # First set the models to ensure they're available for any code that needs them
    from anode import set_inference_model, set_optimization_model
    set_inference_model(args.inference_model)
    set_optimization_model(args.optimization_model)

    # Then load the prompt module
    prompt_spec = importlib.util.spec_from_file_location("prmopt_module", args.prompt_file)
    prompt_module = importlib.util.module_from_spec(prompt_spec)
    prompt_spec.loader.exec_module(prompt_module)
    new_optimize = getattr(prompt_module, "optimize", None)
    print(new_optimize)
    assert callable(new_optimize)
    set_optimize(new_optimize)

    with open(args.prompt_file, "r") as f:
        exec(f.read())

    es = create_engine(f"sqlite:///{args.db_name}")
    SQLModel.metadata.create_all(es)
    set_es(es)

    if args.test:
        test_run(args)
        test_optimize(args)
        exit()

    with S(es) as session:
        session.expire_on_commit = False
        asyncio.run(main(args))