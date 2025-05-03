from aflow import get_graph_from_a_file, set_es
from sqlmodel import Field, Relationship, SQLModel, create_engine, Session as S, select
from data.vizwiz import get_a_random_task
import asyncio
from anode import set_inference_model

es = create_engine(f"sqlite:///db.sqlite")
SQLModel.metadata.create_all(es)
set_es(es)
set_inference_model('qwen-vl-max-latest')

graph = get_graph_from_a_file('seed1.py')
print(asyncio.run(graph.run(get_a_random_task())))
