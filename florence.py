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
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)

    def detect(self, image: Image.Image, objects: List[str], box_threshold: float = 0.3) -> List[Bbox]:
        image = image.convert("RGB")
        text_prompt = ". ".join([obj.strip().lower() for obj in objects]) + "."
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            box_threshold=box_threshold,
            text_threshold=0.25,
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

def get_pipeline_detector(model_name: str = "IDEA-Research/grounding-dino-base"):
    pipe = pipeline("zero-shot-object-detection", model=model_name)

    def detect(image: Image.Image, objects: List[str], box_threshold: float = 0.3) -> List[Bbox]:
        results = pipe(image, candidate_labels=objects, threshold=box_threshold)
        detections = []
        for result in results:
            box = [
                result["box"]["xmin"],
                result["box"]["ymin"],
                result["box"]["xmax"],
                result["box"]["ymax"]
            ]
            detections.append(Bbox(box=box, score=result["score"], label=result["label"]))
        return detections

    return detect

g_dino = G_Dino()
pipeline_detector = get_pipeline_detector()

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    try:
        image = Image.open(image_path)
        objects = ["person", "car", "dog"]

        detections = g_dino.detect(image, objects)
        print(f"Detected {len(detections)} objects using G_Dino class")
        for det in detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")

        result_image = g_dino.draw_boxes(image.copy(), detections)
        result_image.save("result_class.jpg")

        pipeline_detections = pipeline_detector(image, objects)
        print(f"Detected {len(pipeline_detections)} objects using pipeline")
        for det in pipeline_detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")
    except Exception as e:
        print(f"Error: {e}")