from anode import lmm  # a convenient wrapper around lmm api calls, takes in str or Image.Image as args
from PIL import Image

PROMPT1 = """when answering question '{question}', a first step can be outputting a bounding box containing area that is related to this question. can you output the coordinates of a x y x y bounding box between <bbox> and </bbox>,  that contains all relevant information and keeps out irrelevant parts?"""
PROMPT0 = "{question}? let's think step by step and put final answer in curly braces like this: {{final_numeric_answer}}"

def run(image: Image.Image, question: str) -> str:
    ret = lmm(image, PROMPT1.format(question=question))
    bbox_str = ret.split('<bbox>')[1].split('</bbox>')[0]
    bbox = tuple(float(x) for x in bbox_str.strip('()[] ').replace(',', ' ').split())
    image = image.crop(bbox)

    ret = lmm(image, PROMPT0.format(question=question))
    ret = ret.split(r'{')[-1].split(r'}')[0]
    return ret