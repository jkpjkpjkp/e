import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
from typing import List, TypedDict
from PIL import Image, ImageDraw

class Bbox(TypedDict):
    box: List[float]
    score: float
    label: str

class G_Dino:
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)

    def detect(self, image: Image.Image, objects: List[str], box_threshold: float = 0.3, text_threshold: float = 0.25) -> List[Bbox]:
        image = image.convert("RGB")
        text_prompt = ". ".join([obj.strip().lower() for obj in objects]) + "."
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append(Bbox(box=box.tolist(), score=score.item(), label=label))

        return detections

    def draw_boxes(self, image: Image.Image, detections: List[Bbox]) -> Image.Image:
        draw = ImageDraw.Draw(image)
        for detection in detections:
            box = detection["box"]
            label = detection["label"]
            score = detection["score"]
            draw.rectangle(box, outline="red", width=3)
            text = f"{label}: {score:.2f}"
            draw.text((box[0], box[1] - 10), text, fill="red")
        return image

g_dino = G_Dino()