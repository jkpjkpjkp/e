from typing import List
from PIL import Image
from od import Bbox

def run(image: Image.Image, labels: List[str]) -> List[Bbox]:
    """Test function that returns dummy bounding boxes for given labels."""
    detections = []
    for label in labels:
        width, height = image.size
        box_width, box_height = width // 4, height // 4
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2

        detections.append(Bbox(
            box=[x1, y1, x1 + box_width, y1 + box_height],
            score=0.95,
            label=label
        ))

    return detections
