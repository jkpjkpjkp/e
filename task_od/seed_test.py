import torch
from typing import List, Tuple
from PIL import Image
import PIL.Image
from od import Bbox

def run(image: Image.Image, labels: List[str]) -> List[Bbox]:
    """
    A simple test function that returns dummy bounding boxes for the given labels.
    This is used for testing the data loading pipeline without relying on actual models.
    """
    # Create a dummy detection for each label
    detections = []
    for label in labels:
        # Create a dummy bounding box in the center of the image
        width, height = image.size
        box_width = width // 4
        box_height = height // 4
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        detections.append(Bbox(
            box=[x1, y1, x2, y2],
            score=0.95,
            label=label
        ))
    
    return detections
