from sqlmodel import Field, Relationship, SQLModel, create_engine, Session as S, select
from sqlalchemy import Column, func
from sqlalchemy.types import JSON
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
from PIL import Image
from io import BytesIO

from aflow_prompt import WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_OPTIMIZE_GUIDANCE, format_log, format_experience
from anode import custom
import pydantic._internal._model_construction
import base64
import wandb
import argparse


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
        with S(es) as session:
            ret = session.exec(
                select(Run)
                .where(Run.graph_id == self.id)
                .where(Run.task_id == task['id'])
            ).first()
            if ret:
                return ret

        with wandb.init() as wandb_run:
            namespace = {
                '__name__': '__exec__',
                '__package__': None,
            }

            assert task['image']
            assert task['question']
            try:
                exec(self.graph, namespace)
                run = namespace.get('run')
                ret = run(task['image'], task['question'])
            except Exception as e:
                print(f'ERROR Graph.run: {e}')
                ret = Run(
                    graph_id=self.id,
                    task_id=task['id'],
                    log={
                        'input': (task['image'], task['question']),
                        'output': f'ERROR Graph.run: {e}',
                        'correct answer': task['answer'],
                        'wandb': {
                            'run_id': wandb_run.id,
                            'url': wandb_run.url
                        }
                    },
                    output=f'ERROR Graph.run: {e}',
                    score=0,
                )
                with S(es) as session:
                    session.add(ret)
                    session.commit()
                
                raise
            
            score = loss(output=ret, answer=task['answer'])
            
            ret = Run(
                graph_id=self.id,
                task_id=task['id'],
                output=ret,
                log={
                    'input': (task['image'], task['question']),
                    'output': ret,
                    'correct answer': task['answer'],
                    'wandb': {
                        'run_id': wandb_run.id,
                        'url': wandb_run.url
                    }
                },
                score=score,
            )
            with S(es) as session:
                session.expire_on_commit = False
                session.add(ret)
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
        def bytes_to_image(obj):
            if isinstance(obj, str):
                if obj.startswith('abragadoabragado'):
                    return Image.open(BytesIO(base64.b64decode(obj[len('abragadoabragado'):])))
                return obj
            elif isinstance(obj, list):
                return [bytes_to_image(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(bytes_to_image(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: bytes_to_image(v) for k, v in obj.items()}
            else:
                raise TypeError(type(obj))
        return bytes_to_image(self.loglog)
    @log.setter
    def log(self, value):
        def image_to_bytes(obj):
            if isinstance(obj, Image.Image):
                buffered = BytesIO()
                obj.save(buffered, format="PNG")
                return f'abragadoabragado{base64.b64encode(buffered.getvalue()).decode()}'
            elif isinstance(obj, str):
                return obj
            elif isinstance(obj, list):
                return [image_to_bytes(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(image_to_bytes(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: image_to_bytes(v) for k, v in obj.items()}
            else:
                assert issubclass(type(obj), BaseModel) or isinstance(obj, pydantic._internal._model_construction.ModelMetaclass)
                return type(obj).__name__
        self.loglog = image_to_bytes(value)
    



def put(x):
    with S(es) as session:
        session.add(x)
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
    with S(es) as session:
        run_task_ids = session.exec(select(Run.task_id)).all()
        for task_id in get_all_task_ids():
            if task_id not in run_task_ids:
                ret.append(task_id)
    ret = ret[:k]
    with S(es) as session:
        ret.extend(
            session.exec(
                select(Run.task_id).
                group_by(Run.task_id).
                order_by(func.sqrt(
                    func.avg(Run.score * Run.score) - func.avg(Run.score) * func.avg(Run.score)
                ).desc())
                .limit(k - len(ret))
            ).all()
        )
    ret = [get_task_by_id(id) for id in ret]
    return ret[0] if k == 1 else ret


class GraphOp(BaseModel):
    plan: str = Field(description="Thoughts, analysis, of plan on how to improve the agent")
    modification: str = Field(description="Briefly describe the modification made to the agent")
    agent: str = Field(description="The agent code and prompts (`run` function)")

async def main():
    graph = get_graph_from_a_file('seed.py')
    await graph.run(get_high_variance_task())

    strongest = None
    wins = 0
    
    for _ in range(10):
        if wins == 5:
            break
        graphs = get_strongest_graph(3)
        tasks = get_high_variance_task(5)
        result = await asyncio.gather(*[graph.run(task) for graph in graphs for task in tasks])

        graph = get_strongest_graph()
        if graph == strongest:
            wins += 1
        else:
            strongest = graph,
            wins = 0

        run = await graph.run(get_high_variance_task())
        ret = custom(
            WORKFLOW_OPTIMIZE_PROMPT.format(
                agent=graph.graph,
                experience=format_experience(graph),
                score=run.score,
            ),
            *format_log(run.log),
            WORKFLOW_OPTIMIZE_GUIDANCE,
            dna=GraphOp,
        )

        graph = Graph(
            graph=ret.agent,
            father=graph,
            change=ret.modification,
        )
        put(graph)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['zero', 'counts'],
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
        "--seed",
        type=str,
        default="seed.py",
        help="initial graph python file",
    )
    args = parser.parse_args()
    with open(f"{args.dataset}.py", "r") as f:
        exec(f.read())
    
    es = create_engine(f"sqlite:///{args.db_name}")
    SQLModel.metadata.create_all(es)
    asyncio.run(main())
