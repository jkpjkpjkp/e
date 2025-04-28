from lmm import lmm  # a convenient wrapper around lmm api call, takes in as args str or Image.Image
from PIL import Image

PROMPT = """to answer 'How many {label} is visible in the image', a first step can be to split the image in 2 and examine each. output the coordinates of a x y x y bounding box between <bbox> and </bbox>, that is one half of the image we should split into. be careful not to cut through any object of interest"""

def run(image: Image.Image, label: str) -> tuple[float, float, float, float]:
    ret = lmm(image, PROMPT.format(label=label))
    bbox_str = ret.response.split('<bbox>')[1].split('</bbox>')[0]
    if bbox_str.startswith(('(', '[')):
        bbox_str = bbox_str[1:-1]
    bbox = tuple(float(x) for x in bbox_str.replace(',', ' ').split())
    return bbox
