import polars as pl
from PIL import Image

df = pl.read_parquet('dataset_grouped.parquet')

def get_all_task_ids():
    return list(range(len(df)))

def get_task_by_id(id):
    ret = df[id].to_dict()
    ret['image'] = Image.open(ret['image_path'][0])
    ret['label'] = ret['label'][0]
    ret['answer'] = len(ret['annotations'][0])
    ret['id'] = id
    return ret

def loss(output, answer) -> float:
    score = 1 - 4.2 * abs(output - answer) / answer
    score = max(0, score)
    return score