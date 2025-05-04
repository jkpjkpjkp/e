from sqlmodel import Field, Relationship, SQLModel, create_engine, Session as S, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
from PIL import Image
from io import BytesIO

from task_main.prompts import WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_OPTIMIZE_GUIDANCE, format_log, format_experience
from anode import custom
import base64
import argparse
import sys
import random
import os
from util import *
from tqdm import tqdm
import time

image_cache_dir = '/data/image_cache'

es = None
get_all_task_ids = None
get_task_by_id = None

def set_es(_es):
    global es
    es = _es

def set_data(get_all_task_ids_, get_task_by_id_):
    global get_all_task_ids, get_task_by_id
    get_all_task_ids = get_all_task_ids_
    get_task_by_id = get_task_by_id_
    assert get_all_task_ids is not None
    assert get_task_by_id is not None

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
            session.expire_on_commit = False
            return session.exec(
                select(Graph)
                .where(Graph.father_id == self.id)
            ).all()

    @property
    def score(self) -> float:
        with S(es) as session:
            session.expire_on_commit = False
            return (lambda x: sum(x) / len(x))(
                session.exec(
                    select(Run.score)
                    .where(Run.graph_id == self.id)
                ).all()
            )

    async def run(self, task):
        assert isinstance(task, dict)
        assert isinstance(task['image'], Image.Image)
        assert task['question']
        assert task['answer']
        assert task['id'] is not None

        graph_id = self.id

        with S(es) as session:
            session.expire_on_commit = False
            ret = session.exec(
                select(Run)
                .where(Run.graph_id == graph_id)
                .where(Run.task_id == task['id'])
            ).first()
            if ret:
                print(f"Cache hit for graph_id={graph_id}, task_id={task['id']}")
                return ret
            else:
                print(f"Cache miss for graph_id={graph_id}, task_id={task['id']} - creating new run")

        namespace = {
            '__name__': '__exec__',
            '__package__': None,
        }
        keywords = ['lmm']
        trace_log = []
        def trace_function_call(frame, event, arg):
            if event not in ('call', 'return'):
                return None
            if frame.f_code.co_name not in keywords:
                return None

            if event == 'call':
                func_name = frame.f_code.co_name
                args = {}
                for k, v in frame.f_locals.items():
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        args[k] = v
                    elif hasattr(v, '__class__') and v.__class__.__name__ == 'FrameLocalsProxy':
                        continue
                    else:
                        args[k] = f"<{type(v).__name__} object>"

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
                        if isinstance(arg, (str, int, float, bool, list, dict)):
                            trace_log[i]['return'] = arg
                        elif hasattr(arg, 'model_dump') and callable(arg.model_dump):
                            try:
                                trace_log[i]['return'] = f"<{type(arg).__name__}: {str(arg)}>"
                            except:
                                trace_log[i]['return'] = f"<{type(arg).__name__} object>"
                        elif hasattr(arg, '__dict__'):
                            try:
                                trace_log[i]['return'] = f"<{type(arg).__name__}: {str(arg)}>"
                            except:
                                trace_log[i]['return'] = f"<{type(arg).__name__} object>"
                        else:
                            trace_log[i]['return'] = str(arg)
                        break
                else:
                    print(f"No matching call found for {func_name} return")
                return trace_function_call
            return None

        exec(self.graph, namespace)
        run = namespace.get('run')

        sys.settrace(trace_function_call)
        ret = run(task['image'], task['question'])
        sys.settrace(None)

        # Handle ChatCompletionMessage objects
        if hasattr(ret, 'content'):
            ret_content = ret.content
        else:
            ret_content = str(ret)

        score = task['score'](ret_content)

        print(trace_log)
        ret = Run(
            graph_id=self.id,
            task_id=task['id'],
            output=ret_content,
            log=trace_log,
            score=score,
        )
        with S(es) as session:
            session.expire_on_commit = False
            session.expire_on_commit = False
            session.add(ret)
            session.commit()

        return ret

class Run(SQLModel, table=True):
    graph_id: int = Field(primary_key=True, foreign_key="graph.id")
    task_id: str = Field(primary_key=True)
    loglog: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    output: str  # = Field(sa_column=Column(JSON))
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
        session.expire_on_commit = False
        session.add(x)
        session.commit()
        return x

def get_graph_from_a_file(path: str):
    with open(path, "r") as f:
        graph = f.read()
    graph = Graph(graph=graph)
    with S(es) as session:
        session.expire_on_commit = False
        existing = session.exec(select(Graph).where(Graph.graph == graph.graph)).first()
        if existing:
            return existing
        session.add(graph)
        session.commit()
        session.refresh(graph)
    return graph


def get_strongest_graph(k=1):
    with S(es) as session:
        session.expire_on_commit = False
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
        session.expire_on_commit = False
        run_task_ids = session.exec(select(Run.task_id)).all()
        for task_id in all_task_ids:
            if task_id not in run_task_ids:
                ret.append(task_id)
    ret = ret[:k]
    with S(es) as session:
        session.expire_on_commit = False
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


def get_random_task(k=1):
    all_task_ids = get_all_task_ids()
    selected_ids = random.sample(all_task_ids, k)
    ret = [get_task_by_id(id) for id in selected_ids]
    return ret[0] if k == 1 else ret

def get_random_or_high_variance_task(k=1):
    if random.random() < 0.5:
        return get_random_task(k)
    else:
        return get_high_variance_task(k)


