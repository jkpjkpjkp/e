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

from anode import lmm  # a convenient wrapper around lmm api calls, takes in str or Image.Image as args
from PIL import Image

PROMPT0 = "How many {label}s is in the image? let's think step by step and put final answer in curly braces like this: {{final_numeric_answer}}"
PROMPT1 = """when answering question 'How many {label}s is visible in the image', a first step can be outputting a bounding box containing area that is related to this question. can you output the coordinates of a x y x y bounding box between <bbox> and </bbox>,  that contains all relevant information and keeps out irrelevant parts?"""
PROMPT2 = """to answer 'How many {label}s is visible in the image', a first step can be to split the image in 2 and examine each. output the coordinates of a x y x y bounding box between <bbox> and </bbox>, that is one half of the image we should split into. be careful not to cut through any object of interest"""

def run(image: Image.Image, label: str) -> int:
    print(image, label)
    ret = lmm(image, PROMPT0.format(label=label))
    ret = int(ret.split(r'{')[-1].split(r'}')[0])
    if ret <= 4:
        return ret

    ret = lmm(image, PROMPT1.format(label=label))
    bbox_str = ret.split('<bbox>')[1].split('</bbox>')[0]
    bbox = tuple(float(x) for x in bbox_str.strip('()[] ').replace(',', ' ').split())
    print('zoom: ', bbox)
    image = image.crop(bbox)

    ret = lmm(image, PROMPT2.format(label=label))
    bbox_str = ret.split('<bbox>')[1].split('</bbox>')[0]
    bbox1 = tuple(float(x) for x in bbox_str.strip('()[] ').replace(',', ' ').split())
    # bbox2 is the largest rectangle in remains of image after cropping out bbox1
    w, h = image.size
    x1, y1, x2, y2 = bbox1
    candidates = [(0, 0, x1, h), (x2, 0, w, h), (0, 0, w, y1), (0, y2, w, h)]
    bbox2 = max(candidates, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    print('splits: ', bbox1, bbox2)
    return run(image.crop(bbox1), label) + run(image.crop(bbox2), label)

if __name__ == '__main__':
    task = get_task_by_id(88)
    print(run(task['image'], task['label']), task['answer'])
