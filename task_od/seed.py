import torch
from typing import List, Tuple, Dict
from PIL import Image
import PIL.Image
import requests
import base64
from io import BytesIO
from od import grounding_dino, owl_v2, Bbox
from florence import G_Dino

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
    owl_threshold = 0.1
    dino_box_threshold = 0.2
    dino_text_threshold = 0.1

    # Get detections from G_Dino
    g_dino = G_Dino()
    g_dino_detections = g_dino.detect(image, labels, box_threshold=dino_box_threshold)

    # Get detections from OWL
    owl_detections = owl_v2(image, labels, threshold=owl_threshold)[0]

    # If no detections from G_Dino, return owl detections
    if not g_dino_detections:
        print("No detections from G_Dino, using OWL detections only")
        return owl_detections

    # Process G_Dino detections
    trimmed_g_dino_detections = trim_result(g_dino_detections)

    # If no detections from OWL, return trimmed G_Dino detections
    if not owl_detections:
        print("No detections from OWL, using G_Dino detections only")
        return trimmed_g_dino_detections

    # Filter G_Dino detections based on OWL labels
    owl_labels = {x['label'] for x in owl_detections}
    filtered_detections = [x for x in trimmed_g_dino_detections if x['label'] in owl_labels]

    # If no filtered detections, return the combined results from both models
    if not filtered_detections:
        print("No overlapping detections, returning combined results")
        return trimmed_g_dino_detections + owl_detections

    return filtered_detections