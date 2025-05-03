import torch
from typing import List, Tuple
from PIL import Image as Img
import PIL.Image
import requests
import base64
from io import BytesIO
from .dino import draw_boxes, format_detections, get_dino
from .owl import get_owl
from .bbox import Bbox

class ObjectDetectionFactory:
    """Grounding tool interfacing for object detection."""

    @staticmethod
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

    @staticmethod
    def trim_result(detections: List[Bbox]) -> List[Bbox]:
        """Group detections by label and trim each group."""
        unique_labels = {bbox['label'] for bbox in detections}
        final_detections = []
        for label in unique_labels:
            label_detections = [d for d in detections if d['label'] == label]
            trimmed = ObjectDetectionFactory.box_trim(label_detections)
            final_detections.extend(trimmed)
        return final_detections

    @staticmethod
    def _run(image: Img, texts: List[str], owl_threshold=0.1, dino_box_threshold=0.2, dino_text_threshold=0.1) -> List[Bbox]:
        """Run detection using both servers and combine results with adjustable thresholds."""
        if not isinstance(image, Img):
            image = PIL.Image.fromarray(image)
        # Ensure image is in RGB format for consistent processing
        image = image.convert('RGB')

        owl_detections = get_owl()(image, texts, threshold=owl_threshold)[2]
        dino_detections = get_dino()(image, texts, box_threshold=dino_box_threshold, text_threshold=dino_text_threshold)[2]
        trimmed_dino_detections = ObjectDetectionFactory.trim_result(dino_detections)

        owl_labels = {x['label'] for x in owl_detections}
        filtered_detections = [x for x in trimmed_dino_detections if x['label'] in owl_labels]

        return filtered_detections

def run(image: Img, texts: List[str]) -> Tuple[Img, str, List[Bbox]]:
    """Perform object detection on an image with given texts and return annotated results."""
    owl_threshold = 0.1
    dino_box_threshold = 0.2
    dino_text_threshold = 0.1
    detections = ObjectDetectionFactory._run(image, texts, owl_threshold, dino_box_threshold, dino_text_threshold)
    image_with_boxes = draw_boxes(image.copy(), detections)
    detection_details = format_detections(detections)
    return image_with_boxes, detection_details, detections