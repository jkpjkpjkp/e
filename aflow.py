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
        while True:
            with S(es) as session:
                session.expire_on_commit = False
                graph = session.merge(graph)
                correct, incorrect = get_correct_incorrect(graph.runs)
                print(f"Current runs: {len(graph.runs)}, Correct: {len(correct)}, Incorrect: {len(incorrect)}")
            if min(len(correct), len(incorrect)) >= 5:
                break

            tasks = get_random_or_high_variance_task(args.experiment_strategy)
            print(f"Running additional tasks: {[task['id'] for task in tasks]}")
            await asyncio.gather(*[graph.run(task) for task in tasks])
            import time
            time.sleep(1)


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

        graph = optimize(graph)
        tasks = get_high_variance_task(args.experiment_strategy)
        # Run final tasks for the new graph
        await asyncio.gather(*[graph.run(task) for task in tasks])



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
        help="Dataset file",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Prompt file",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        help="Database name",
    )
    parser.add_argument(
        "--inference-model",
        type=str,
        default="plus",
        help="Model to use for inference when calling lmm()",
    )
    parser.add_argument(
        "--optimization-model",
        type=str,
        default="max",
        help="Model to use for optimization",
    )
    parser.add_argument(
        "--experiment-strategy",
        type=str,
        default="5",
        help="how many tasks to run, 'all' for full eval",
    )
    args = parser.parse_args()

    model_map = {
        'plus': 'qwen-vl-plus-latest',
        'max': 'qwen-vl-max-latest',
        'claude': 'claude-3-7-sonnet-20250219',
    }
    if args.inference_model in model_map:
        args.inference_model = model_map[args.inference_model]
    if args.optimization_model in model_map:
        args.optimization_model = model_map[args.optimization_model]
    
    if not args.seed_file:
        args.seed_file = f"{args.folder}/seed.py"
    if not args.data_factory:
        args.data_factory = f"{args.folder}/data.py"
    if not args.prompt_file:
        args.prompt_file = f"{args.folder}/prompts.py"
    if not args.db_name:
        args.db_name = f"{args.folder}/db.sqlite"
    
    data_file_path = os.path.abspath(args.data_factory)

    spec = importlib.util.spec_from_file_location("data_module", data_file_path)
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    new_get_all_task_ids = getattr(data_module, "get_all_task_ids", None)
    new_get_task_by_id = getattr(data_module, "get_task_by_id", None)

    # Step 5: Validate that the function exists and is callable
    assert callable(new_get_all_task_ids) and callable(new_get_task_by_id)
    with open(args.prompt_file, "r") as f:
        exec(f.read())
    
    set_data(new_get_all_task_ids, new_get_task_by_id)

    from anode import set_inference_model, set_optimization_model
    set_inference_model(args.inference_model)
    set_optimization_model(args.optimization_model)

    es = create_engine(f"sqlite:///{args.db_name}")
    SQLModel.metadata.create_all(es)
    set_es(es)

    with S(es) as session:
        session.expire_on_commit = False
        asyncio.run(main(args))