import polars as pl
from PIL import Image
from anode import LLM
import io
import random

df = pl.read_parquet('/data/viswiz/val-0000*of-00005-*.parquet')
df = df.filter(pl.col('category') != 'unanswerable')

print(df.head())
print(df.columns)

def get_all_task_ids():
    return list(df['question_id'])

def get_task_by_id(id):
    row = df.filter(pl.col('question_id') == id).row(0, named=True)
    row = dict(row)
    row['id'] = id
    row['answer'] = row['answers']
    row['image'] = Image.open(io.BytesIO(row['image']['bytes']))
    row['loss'] = lambda x: loss(output=x, answer=row['answer'])
    return row


def llm_as_judge(output, answer):
    llm = LLM('deepseek-chat')
    
    prompt = f"""You are judging the correctness of a response, based on ground truth answers.

output: {output}

list of ground truth answers: {answer}

Think step by step. 
finish your thought process, and return {{1}} if the prediction is correct, {{0}} if incorrect. 
remember to add the curly braces around your score. 
."""
    try:
        response = llm.aask(prompt)
        score = response.split('{')[1].split('}')[0]
        return float(score.strip())
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0

def loss(output, answer) -> float:
    answers = answer
    answers = [x for x in answers if x != 'unanswerable']
    return llm_as_judge(output, answers)

def get_a_random_task():
    return get_task_by_id(random.choice(get_all_task_ids()))
