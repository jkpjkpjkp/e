import torch
from typing import List, Tuple
from PIL import Image
import PIL.Image
import requests
import base64
from io import BytesIO
from od import grounding_dino, owl_v2, Bbox

def box_trim(detections: List[Bbox]) -> List[Bbox]:
    """Trim overlapping detections based on occlusion threshold."""
    occlusion_threshold = 0.3
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    kept_detections = []

    def area(box: List[float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x1 >= x2 or y1 >= y2:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    for candidate in sorted_detections:
        keep = True
        for accepted_bbox in kept_detections:
            inter_area = intersection_area(candidate['box'], accepted_bbox['box'])
            accepted_area = area(accepted_bbox['box'])
            if accepted_area == 0:
                continue
            ioa = inter_area / accepted_area
            if ioa >= occlusion_threshold:
                keep = False
                break
        if keep:
            kept_detections.append(candidate)
    return kept_detections

def trim_result(detections: List[Bbox]) -> List[Bbox]:
    """Group detections by label and trim each group."""
    unique_labels = {bbox['label'] for bbox in detections}
    final_detections = []
    for label in unique_labels:
        label_detections = [d for d in detections if d['label'] == label]
        trimmed = box_trim(label_detections)
        final_detections.extend(trimmed)
    return final_detections


def run(image: Image.Image, labels: List[str]) -> List[Bbox]:
    """Perform object detection on an image with given texts and return annotated results."""
    owl_threshold = 0.1
    dino_box_threshold = 0.2
    dino_text_threshold = 0.1
    owl_detections = owl_v2(image, labels, threshold=owl_threshold)[0]
    dino_detections = grounding_dino(image, labels, box_threshold=dino_box_threshold, text_threshold=dino_text_threshold)[0]
    trimmed_dino_detections = trim_result(dino_detections)

    owl_labels = {x['label'] for x in owl_detections}
    filtered_detections = [x for x in trimmed_dino_detections if x['label'] in owl_labels]

    return filtered_detections