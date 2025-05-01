from gd import get_dino
from PIL import Image
from typing import List
from anode import lmm  # noqa: F401

def grounding_dino(image: Image.Image, objects: List[str], box_threshold=0.2, text_threshold=0.15):
    detector = get_dino(max_parallel=1)
    return detector(image, objects, box_threshold, text_threshold)