import polars as pl
from PIL import Image
import random

df = pl.read_parquet('dataset_grouped.parquet')

def get_all_task_ids():
    return list(map(str, range(len(df))))


def IoU(output, answer):
    x1=max(output[0], answer[0])
    y1=max(output[1], answer[1])
    x2=min(output[2], answer[2])
    y2=min(output[3], answer[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (output[2] - output[0]) * (output[3] - output[1])
        area2 = (answer[2] - answer[0]) * (answer[3] - answer[1])
        union = area1 + area2 - intersection
        return intersection / union


def get_task_by_id(id):
    ret = df[id].to_dict()
    ret['image'] = Image.open(ret['image_path'][0])
    ret['question'] = f"What is the smallest bounding box containing ALL {ret['label'][0]}s in the image? "

    min_x, min_y, max_x, max_y = 1000000, 1000000, 0, 0
    for bbox in ret['annotations'][0]:
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[0] + bbox[2])
        max_y = max(max_y, bbox[1] + bbox[3])

    ret['answer'] = (min_x, min_y, max_x, max_y)
    ret['id'] = id

    ret['score'] = lambda output: IoU(output, ret['answer'])
    ret['id'] = id
    return ret

def get_dummy_task():
    # a white 540x540 image
    image = Image.new('RGB', (540, 540), (255, 255, 255))
    # a random black rectangle inside, prompt to bbox that rectangle
    x1, y1, x2, y2 = random.randint(0, 540), random.randint(0, 540), random.randint(0, 540), random.randint(0, 540)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    image.paste((Image.new('RGB', (x2 - x1, y2 - y1), (0, 0, 0)), (x1, y1)))
    return {
        'image': image,
        'question': 'What is the bounding box of the black rectangle in the image? ',
        'answer': (x1, y1, x2, y2),
        'id': f'dummy_{x1}_{y1}_{x2}_{y2}',
        'score': lambda output: IoU(output,  (x1, y1, x2, y2)),
    }

WORKFLOW_OPTIMIZE_PROMPT = """We need to implement the function `run`, which takes in an image and a question (which asks for a bbox), and returns the bbox. Please reconstruct and optimize the function.

You have to extract a numeric tuple bbox from lmm response str, which involves designing a structured output scheme and parsing it. Since the lmm is a small one, you may need to add few shots.
Additionally, it is important that your code does not fallback to dummy return in any situation. raise as soon as possible to catch any errors, including formatting errors. 

Output the modified code under the same setting and function name (run).

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