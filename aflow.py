import uuid
from sqlmodel import Field, Relationship, SQLModel, createengine, Session as S, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any, Optional, Callable
import io
import sys
import functools
import re
from pydantic import BaseModel
import datetime
import numpy as np
import asyncio

from data.data import get_task_by_id, get_all_task_ids
from aflow_prompt import WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_INPUT
import openai

before = """
import openai
model = 'claude-3-7-sonnet-20250219'

def has_model_param(func):
    try:
        sig = inspect.signature(func)
        return 'model' in sig.parameters
    except ValueError:
        return False

class ModelWrapper:
    def __init__(self, wrapped, model=model):
        self.wrapped = wrapped
        self.model = model
        self.log = []

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)

        if name != 'create':
            return ModelWrapper(attr, self.model)
        assert callable(attr) and has_model_param(attr)

        def wrapper(*args, **kwargs):
            kwargs['model'] = self.model
            response = attr(*args, **kwargs)
            kwargs.pop('model')
            self.log.append({
                'message': (args, kwargs),
                'response': response,
            })
            return response
        return wrapper

client = ModelWrapper(openai)
"""

after = """
"""

db_name = "one.sqlite"


class Graph(SQLModel, table=True):
    id: int = Field(primary_key=True)
    graph: str
    prompt: str
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
        with S(e) as session:
            return session.exec(
                select(Graph)
                .where(Graph.father_id == self.id)
            ).all()

    def average_score(self) -> float:
        with S(e) as session:
            return (lambda x: sum(x) / len(x))(
                session.exec(
                    select(Run.score)
                    .where(Run.graph_id == self.id)
                ).all()
            )

    async def run(self, task):
        with S(e) as session:
            ret = session.exec(
                select(Run.output, Run.score)
                .where(Run.graph_id == self.id and Run.task_id == task['id'])
            ).first()
            if ret:
                return ret

        namespace = {
            '__name__': '__exec__',
            '__package__': None,
        }

        exec(before, namespace)
        client = namespace.get('client')

        try:
            exec(self.graph, namespace)
            exec(after, namespace)
            run = namespace.get('run')
            ret = run(task['image'], task['label'])
        except Exception as e:
            print(f'ERROR Graph.run: {e}')
            with S(e) as session:
                session.add(
                    Run(
                        graph_id=self.id,
                        task_id=task['id'],
                        log=client.log,
                        output=f'ERROR Graph.run: {e}',
                        score=0,
                    )
                )
                session.commit()
            return f'ERROR Graph.run: {e}', 0
        
        score = IoU_xyxy(ret, task['answer'])
        
        with S(e) as session:
            session.add(
                Run(
                    graph_id=self.id,
                    task_id=task['id'],
                    log=client.log,
                    output=ret,
                    score=score,
                )
            )
            session.commit()
        
        return ret, score

def A_xyxy(x):
    return (x[2] - x[0]) * (x[3] - x[1])

def IoU_xyxy(a, b):
    I = max((min(a[2], b[2]) - max(a[0], b[0])), 0) * max((min(a[3], b[3]) - max(a[1], b[1])), 0)
    return I / (A_xyxy(a) + A_xyxy(b) - I)


class Run(SQLModel, table=True):
    graph_id: int = Field(primary_key=True, foreign_key="graph.id")
    task_id: str = Field(primary_key=True)
    log: Dict[str, Any] = Field(sa_column=Column(JSON))
    output: tuple[float]
    score: float
    graph: Graph = Relationship(back_populates="runs")

    @property
    def task(self):
        return get_task_by_id(self.task_id)
    

e = createengine(f"sqlite:///{db_name}")
SQLModel.metadata.create_all(e)


def put(x):
    with S(e) as session:
        session.add(x)
        session.commit()
        return x

def get_graph_from_a_file(path: str):
    with open(path, "r") as f:
        graph = f.read()
    graph = Graph(graph=graph)
    with S(e) as session:
        session.add(graph)
        session.commit()
        session.refresh(graph)
    return graph


def get_strongest_graph(k=1):
    with S(e) as session:
        stmt_zero_runs = select(Graph).where(~Graph.runs.any()).limit(k)
        graphs_with_zero_runs = session.exec(stmt_zero_runs).all()
        if len(graphs_with_zero_runs) >= k:
            return graphs_with_zero_runs[:k]
        remaining = k - len(graphs_with_zero_runs)
        stmt_with_runs = (
            select(Graph)
            .join(Run)
            .group_by(Graph.id)
            .order_by(func.avg(Run.score).desc())
            .limit(remaining)
        )
        graphs_with_runs = session.exec(stmt_with_runs).all()
        ret = graphs_with_zero_runs + graphs_with_runs
        return ret[0] if k == 1 else ret


def get_high_variance_task(k=1):
    ret = []
    with S(e) as session:
        run_task_ids = session.exec(select(Run.task_id)).all()
        for task_id in get_all_task_ids():
            if task_id not in run_task_ids:
                ret.append(task_id)
    if len(ret) >= k:
        return ret[:k]
    with S(e) as session:
        ret.extend(
            session.exec(
                select(Run.task_id).
                group_by(Run.task_id).
                order_by(func.std(Run.score).desc())
                .limit(k - len(ret))
            ).all()
        )
    return ret[0] if k == 1 else ret


async def main():
    get_graph_from_a_file('seed.py')
    for _ in range(10):
        graphs = get_strongest_graph(3)
        tasks = get_high_variance_task(10)
        
        await asyncio.gather([graph.run(task) for graph in graphs for task in tasks])
        graph = get_strongest_graph()

        response = 
