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
def format_experience(graph):
    failures = [x for x in graph.children if x.score <= graph.score]
    successes = [x for x in graph.children if x.score > graph.score]
    experience = f"Original Score: {graph.score}\n"
    experience += "Some conclusions drawn from past optimization experience:\n\n"
    for failure in failures:
        experience += f"-Absolutely prohibit {failure.modification} (Score: {failure.score})\n"
    for success in successes:
        experience += f"-Absolutely prohibit {success.modification} \n"
    experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax or modifying the prompt."
    return experience

def format_lmm_io_log(log):
    ret = ["Here are all lmm inputs and responses in a given run:\n",]
    for x in log:
        ret.append("\n---------\nInput: ")
        messages = x['message']['args']
        if isinstance(messages, str):
            ret.append(messages)
        else:
            for message in messages:
                if isinstance(message, str):
                    ret.append(message)
                else:
                    assert isinstance(message, Image.Image)
                    ret.append(message.thumbnail((100, 100)))
        if x['message']['kwargs']:
            ret.append(str(x['message']['kwargs']))
        ret.append("\nResponse: ")
        ret.append(x['response'])
    return ret

def format_log(log):
    ret = str(log)
    assert len(ret) < 5000
    return ret

WORKFLOW_OPTIMIZE_PROMPT = """We are designing an agent that can answer visual question answering (VQA) questions.
We need to implement the function run, which takes in an image and a question and returns the answer. Please reconstruct and optimize the function. You can add, modify, or delete functions, parameters, or prompts. Ensure the code you provide is complete and correct, including necessary imports, except for the lmm method, which is a convenient wrapper around a large multimodal model inference. The lmm method takes in any number of str or Image.Image args.

When optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider using Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical representation.

Output the modified code under the same setting and function name (run).

Complex agents may yield better results, but take into consideration the LLM's limited capabilities and potential information loss. It's crucial to include necessary context.

Here is a graph that performed excellently in a previous iteration for VQA. You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <score>{score}</score>
    <graph>{graph}</graph>
    <operator_description>{operator_description}</operator_description>
</sample>

Additionally, here are some sample image-question-answer triples that this graph got correctly: {correct_qa}

Here are some sample image-question-output-answer quadruples where the graph produced wrong outputs: {wrong_qa}

Below is a detailed log of a run with this graph that ended in wrong answer:
{log}
"""

WORKFLOW_OPTIMIZE_GUIDANCE="""
First, analyze the trace, brainstorm, and propose optimization ideas. **Only one detail should be modified**, and **no more than 5 lines of code should be changed**—extensive modifications are strictly prohibited to maintain project focus! Simplifying code by removing unnecessary steps is often highly effective. When adding new functionalities to the graph, ensure necessary libraries or modules are imported, including importing operators from `op`.
it is encouraged that you use cropping on the image to focus on important area. 
you can prompt the lmm to return specific xml fields, and use `re` to parse it. this way you can get typed return fields, for example x y x y bounding box for cropping or highlighting or analysis. 
"""

# OPERATOR_="{id}. {operator_name}: {desc}, with interface {interface}. \n"
def optimize(graph):

    class GraphOp(BaseModel):
        plan: str = Field(description="Thoughts, analysis, of plan on how to improve the graph")
        modification: str = Field(description="Briefly describe the modification made to the graph")
        graph: str = Field(description="The graph (`run` function)")

    with S(es) as session:
        session.expire_on_commit = False
        graph = session.merge(graph)
        correct_runs, wrong_runs = get_correct_incorrect(graph.runs)

    ret = custom(
        WORKFLOW_OPTIMIZE_PROMPT.format(
            graph=graph.graph,
            experience=format_experience(graph),
            score=graph.score,
            correct_qa='\n'.join(map(lambda run: f"question: {run.task['question']}, answer: {run.task['answer']} ", correct_runs[:5])),
            wrong_qa='\n'.join(map(lambda run: f"question: {run.task['question']}, output(wrong): {run.output}, answer: {run.task['answer']} ", wrong_runs[:5])),
            log=format_log(wrong_runs[-1].log),
            operator_description=format,
        ),
        WORKFLOW_OPTIMIZE_GUIDANCE,
        dna=GraphOp,
    )

    graph = Graph(
        graph=ret.graph,
        father=graph,
        change=ret.modification,
    )
    return put(graph)


[{
    "type": "function_call",
    "id": "fc_12345xyz",
    "call_id": "call_12345xyz",
    "name": "get_weather",
    "arguments": "{\"location\":\"Paris, France\"}"
}]