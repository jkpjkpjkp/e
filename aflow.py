from sqlmodel import Field, Relationship, SQLModel, create_engine, Session as S, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
from PIL import Image
from io import BytesIO

from aflow_prompt import WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_OPTIMIZE_GUIDANCE, OPERATOR_DESCRIPTION, format_log, format_experience
from anode import custom
import base64
import argparse
import sys
import random
import os
from util import *
from tqdm import tqdm

image_cache_dir = '/data/image_cache'

es = None

def set_es(_es):
    global es
    es = _es

class Graph(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph: str
    father_id: Optional[int] = Field(default=None, foreign_key="graph.id")
    change: Optional[str] = Field(default=None)
    runs: list["Run"] = Relationship(back_populates="graph")

    @property
    def father(self):
        return Graph.get(self.father_id)
    @father.setter
    def father(self, value):
        self.father_id = value.id
    @property
    def children(self):
        with S(es) as session:
            return session.exec(
                select(Graph)
                .where(Graph.father_id == self.id)
            ).all()

    @property
    def score(self) -> float:
        with S(es) as session:
            return (lambda x: sum(x) / len(x))(
                session.exec(
                    select(Run.score)
                    .where(Run.graph_id == self.id)
                ).all()
            )

    async def run(self, task):
        assert isinstance(task, dict)
        # print(task)
        assert isinstance(task['image'], Image.Image)
        assert isinstance(task['question'], str)
        assert task['answer']
        assert task['id']

        with S(es) as session:
            ret = session.exec(
                select(Run)
                .where(Run.graph_id == self.id)
                .where(Run.task_id == task['id'])
            ).first()
            if ret:
                print(f"Cache hit for graph_id={self.id}, task_id={task['id']}")
                return ret
            else:
                print(f"Cache miss for graph_id={self.id}, task_id={task['id']} - creating new run")

        namespace = {
            '__name__': '__exec__',
            '__package__': None,
        }

        keywords = self.graph.split()
        print(keywords)
        trace_log = []
        def trace_function_call(frame, event, arg):
            if event not in ('call', 'return'):
                return None
            if frame.f_code.co_name not in keywords:
                return None

            if event == 'call':
                func_name = frame.f_code.co_name
                # Handle FrameLocalsProxy objects
                try:
                    # Copy only serializable args
                    args = {}
                    for k, v in frame.f_locals.items():
                        if isinstance(v, (str, int, float, bool, list, dict)):
                            args[k] = v
                        elif hasattr(v, '__class__') and v.__class__.__name__ == 'FrameLocalsProxy':
                            # Skip FrameLocalsProxy objects
                            continue
                        else:
                            # For other complex objects, store their type information
                            args[k] = f"<{type(v).__name__} object>"
                except Exception as e:
                    print(f"Error serializing args: {e}")
                    args = {"error": f"Failed to serialize args: {str(e)}"}

                trace_log.append({
                    'func_name': func_name,
                    'args': args,
                    'frame_id': id(frame)
                })
                return trace_function_call

            elif event == 'return':
                func_name = frame.f_code.co_name
                for i in reversed(range(len(trace_log))):
                    call = trace_log[i]
                    if call['func_name'] == func_name and call.get('frame_id') == id(frame):
                        # Handle different return value types
                        if isinstance(arg, (str, int, float, bool, list, dict)):
                            trace_log[i]['return'] = arg
                        elif hasattr(arg, 'model_dump') and callable(arg.model_dump):
                            # Handle Pydantic models (like OpenAI API responses)
                            try:
                                trace_log[i]['return'] = f"<{type(arg).__name__}: {str(arg)}>"
                            except:
                                trace_log[i]['return'] = f"<{type(arg).__name__} object>"
                        elif hasattr(arg, '__dict__'):
                            # For objects with __dict__, create a simplified representation
                            try:
                                trace_log[i]['return'] = f"<{type(arg).__name__}: {str(arg)}>"
                            except:
                                trace_log[i]['return'] = f"<{type(arg).__name__} object>"
                        else:
                            # For other objects, just store their string representation
                            trace_log[i]['return'] = str(arg)
                        break
                else:
                    print(f"No matching call found for {func_name} return")
                return trace_function_call
            return None

        try:
            exec(self.graph, namespace)
            run = namespace.get('run')

            sys.settrace(trace_function_call)
            ret = run(task['image'], task['question'])
            sys.settrace(None)
        except Exception as e:
            print(f'ERROR Graph.run: {e}')
            # Create input tuple based on whether question is present
            input_data = (task['image'],) if 'question' not in task or not task['question'] else (task['image'], task['question'])

            ret = Run(
                graph_id=self.id,
                task_id=task['id'],
                log='ERROR Graph.run: {e}',
                output=f'ERROR Graph.run: {e}',
                score=0,
            )
            with S(es) as session:
                session.add(ret)
                try:
                    session.flush()  # Flush explicitly to catch the error before commit
                except Exception as e:
                    print("Error during flush. Inspecting new objects:")
                    for obj in session.new:  # New objects to be inserted
                        print(f"Object: {obj}")
                        for attr, value in obj.__dict__.items():
                            if attr != '_sa_instance_state':  # Ignore SQLAlchemy internal state
                                print(f"  {attr}: Type={type(value)}, Value={value}")
                    raise
                session.commit()

            raise

        score = task['loss'](ret)

        # Create input tuple based on whether question is present
        input_data = (task['image'],) if 'question' not in task or not task['question'] else (task['image'], task['question'])

        print(trace_log)
        ret = Run(
            graph_id=self.id,
            task_id=task['id'],
            output=ret,
            log=trace_log,
            score=score,
        )
        with S(es) as session:
            session.expire_on_commit = False
            session.add(ret)
            try:
                session.flush()  # Flush explicitly to catch the error before commit
            except Exception as e:
                print("Error during flush. Inspecting new objects:")
                for obj in session.new:  # New objects to be inserted
                    print(f"Object: {obj}")
                    for attr, value in obj.__dict__.items():
                        if attr != '_sa_instance_state':  # Ignore SQLAlchemy internal state
                            print(f"  {attr}: Type={type(value)}, Value={value}")
                raise
            session.commit()

        return ret

class Run(SQLModel, table=True):
    graph_id: int = Field(primary_key=True, foreign_key="graph.id")
    task_id: str = Field(primary_key=True)
    loglog: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    output: str
    score: float
    graph: Graph = Relationship(back_populates="runs")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = kwargs['log']

    @property
    def task(self):
        return get_task_by_id(self.task_id)

    @property
    def log(self):
        return find_and_convert(str, unit_str_to_img)(self.loglog)

    @log.setter
    def log(self, value):
        self.loglog = find_and_convert(Image.Image, img_to_str)(value)


def put(x):
    with S(es) as session:
        session.add(x)
        try:
            session.flush()  # Flush explicitly to catch the error before commit
        except Exception as e:
            print("Error during flush. Inspecting new objects:")
            for obj in session.new:  # New objects to be inserted
                print(f"Object: {obj}")
                for attr, value in obj.__dict__.items():
                    if attr != '_sa_instance_state':  # Ignore SQLAlchemy internal state
                        print(f"  {attr}: Type={type(value)}, Value={value}")
            raise
        session.commit()
        return x

def get_graph_from_a_file(path: str):
    with open(path, "r") as f:
        graph = f.read()
    graph = Graph(graph=graph)
    with S(es) as session:
        existing = session.exec(select(Graph).where(Graph.graph == graph.graph)).first()
        if existing:
            return existing
        session.add(graph)
        try:
            session.flush()  # Flush explicitly to catch the error before commit
        except Exception as e:
            print("Error during flush. Inspecting new objects:")
            for obj in session.new:  # New objects to be inserted
                print(f"Object: {obj}")
                for attr, value in obj.__dict__.items():
                    if attr != '_sa_instance_state':  # Ignore SQLAlchemy internal state
                        print(f"  {attr}: Type={type(value)}, Value={value}")
            raise
        session.commit()
        session.refresh(graph)
    return graph


def get_strongest_graph(k=1):
    with S(es) as session:
        stmt_with_runs = (
            select(Graph)
            .join(Run)
            .group_by(Graph.id)
            .order_by(func.avg(Run.score).desc())
            .limit(k)
        )
        ret = session.exec(stmt_with_runs).all()
        return ret[0] if k == 1 else ret


def get_high_variance_task(k=1):
    ret = []
    all_task_ids = get_all_task_ids()
    with S(es) as session:
        run_task_ids = session.exec(select(Run.task_id)).all()
        for task_id in all_task_ids:
            if task_id not in run_task_ids:
                ret.append(task_id)
    ret = ret[:k]
    with S(es) as session:
        high_variance_tasks = session.exec(
            select(Run.task_id).
            group_by(Run.task_id).
            order_by(func.sqrt(
                func.avg(Run.score * Run.score) - func.avg(Run.score) * func.avg(Run.score)
            ).desc())
            .limit(k - len(ret))
        ).all()
        ret.extend(high_variance_tasks)
    ret = [get_task_by_id(id) for id in ret]
    return ret[0] if k == 1 else ret

def get_random_or_high_variance_task(k=1):
    if random.random() < 0.1:
        print("Selecting random tasks")
        all_task_ids = get_all_task_ids()
        selected_ids = random.sample(all_task_ids, k)
        ret = [get_task_by_id(id) for id in selected_ids]
        return ret[0] if k == 1 else ret
    else:
        return get_high_variance_task(k)

def get_correct_incorrect(runs):
    correct = []
    incorrect = []
    for run in runs:
        if run.score > 0.9:
            correct.append(run)
        else:
            incorrect.append(run)
    return correct, incorrect


async def main():
    global args

    graph = get_graph_from_a_file(args.seed_file)
    await graph.run(get_high_variance_task())

    strongest = None
    wins = 0
    best_score = -10000
    best_for = 0

    graphs = get_strongest_graph(100)
    # Run initial tasks for each graph
    await asyncio.gather(*[graph.run(get_random_or_high_variance_task()) for graph in graphs])

    for _ in tqdm(range(100)):
        if wins >= 5 or best_for >= 5:
            break

        with S(es) as session:
            assert set(session.exec(
                select(Run.graph_id).group_by(Run.graph_id)
            ).all()) == set(
                session.exec(
                select(Graph.id)
            ).all())

        graphs = get_strongest_graph(3)
        tasks = get_high_variance_task(5)
        # Run tasks for each graph
        await asyncio.gather(*[graph.run(task) for graph in graphs for task in tasks])

        graph = get_strongest_graph()
        while True:
            correct, incorrect = get_correct_incorrect(graph.runs)
            print(f"Current runs: {len(graph.runs)}, Correct: {len(correct)}, Incorrect: {len(incorrect)}")
            if min(len(correct), len(incorrect)) >= 5:
                break

            # Fix: map() returns an iterator, not a list of coroutines
            # Convert to a list of coroutines that asyncio.gather can unpack
            tasks = get_random_or_high_variance_task(5)
            print(f"Running additional tasks: {[task['id'] for task in tasks]}")
            await asyncio.gather(*[graph.run(task) for task in tasks])


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

        class GraphOp(BaseModel):
            plan: str = Field(description="Thoughts, analysis, of plan on how to improve the agent")
            modification: str = Field(description="Briefly describe the modification made to the agent")
            agent: str = Field(description="The agent code and prompts (`run` function)")

        correct_runs, wrong_runs = get_correct_incorrect(graph.runs)

        ret = custom(
            WORKFLOW_OPTIMIZE_PROMPT.format(
                agent=graph.graph,
                experience=format_experience(graph),
                score=graph.score,
                correct_qa='\n'.join(map(lambda x: f"question: {x.task['question']}, answer: {x.task['answer']} ", correct_runs[:5])),
                wrong_qa='\n'.join(map(lambda x: f"question: {x.task['question']}, output(wrong): {x.output}, answer: {x.task['answer']} ", wrong_runs[:5])),
                log=format_log(wrong_runs[-1].log),
                operator_description=OPERATOR_DESCRIPTION,
            ),
            WORKFLOW_OPTIMIZE_GUIDANCE,
            dna=GraphOp,
        )

        graph = Graph(
            graph=ret.agent,
            father=graph,
            change=ret.modification,
        )
        put(graph)
        tasks = get_high_variance_task(5)
        # Run final tasks for the new graph
        await asyncio.gather(*[graph.run(task) for task in tasks])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['zero', 'counts', 'viswiz'],
        required=True,
        help="Dataset type",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="db.sqlite",
        help="Optimized result save db",
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="seed1.py",
        help="initial graph python file",
    )
    parser.add_argument(
        "--inference_model",
        type=str,
        default="qwen-vl-plus-latest",
        help="Model to use for inference when calling lmm()",
    )
    parser.add_argument(
        "--optimization_model",
        type=str,
        default="qwen-vl-max-latest",
        help="Model to use for optimization",
    )
    args = parser.parse_args()
    with open(f"{args.dataset}.py", "r") as f:
        exec(f.read())

    # Set global variables for models using the setter functions from anode
    from anode import set_inference_model, set_optimization_model
    set_inference_model(args.inference_model)
    set_optimization_model(args.optimization_model)

    es = create_engine(f"sqlite:///{args.db_name}")
    SQLModel.metadata.create_all(es)
    asyncio.run(main())
