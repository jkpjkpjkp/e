from gd import get_dino, plot_bounding_boxes
from PIL import Image
from typing import List
from anode import lmm  # noqa: F401

def grounding_dino(image: Image.Image, objects: List[str], box_threshold=0.2, text_threshold=0.15):
    image = image.copy()
    detector = get_dino(max_parallel=1)
    bboxes = detector(image, objects, box_threshold, text_threshold)
    return bboxes, plot_bounding_boxes(image, bboxes)

def test_grounding_dino():
    from viswiz import get_all_task_ids, get_task_by_id
    task = get_task_by_id(get_all_task_ids()[0])
    bbox, img = grounding_dino(task['image'], task['answer'])
    img.show()

if __name__ == '__main__':
    test_grounding_dino()