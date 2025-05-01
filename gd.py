import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, TypedDict, Tuple
from PIL.Image import Image as Img
from PIL import Image, ImageDraw, ImageFont
import threading

class Bbox(TypedDict):
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str

class DinoObjectDetectionFactory:
    def __init__(self, max_parallel=1):
        """Initialize the factory with eager loading of models and concurrency control."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load processor and model eagerly during initialization
        self.gd_processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
        self.semaphore = threading.Semaphore(max_parallel)

    def _run(self, image: Image.Image, texts: List[str], box_threshold=0.2, text_threshold=0.1) -> List[Bbox]:
        """Detect objects in an image using Grounding DINO with concurrency control."""
        if not texts or not image:
            raise ValueError('Valid image and at least one text description required')

        with self.semaphore:  # Controls parallelism
            image = image.convert('RGB')
            text = '. '.join(text.strip().lower() for text in texts) + '.'
            inputs = self.gd_processor(images=image, text=text, return_tensors='pt').to(self.device)
            self.gd_model.to(self.device)
            with torch.no_grad():
                outputs = self.gd_model(**inputs)
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs['input_ids'], box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[image.size[::-1]]
            )[0]
            return [Bbox(box=box.tolist(), score=score.item(), label=label)
                    for box, score, label in zip(results['boxes'], results['scores'], results['labels'])]

_dino_factory = None
_dino_lock = threading.Lock()

def plot_bounding_boxes(img: Image.Image, bounding_boxes: List[Bbox]):
    """Plots bounding boxes on an image with markers for each a name, using different colors. """

    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ]

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    for i, bounding_box in enumerate(bounding_boxes):
        color = colors[i % len(colors)]

        abs_x1 = int(bounding_box["box"][0])
        abs_y1 = int(bounding_box["box"][1])
        abs_x2 = int(bounding_box["box"][2])
        abs_y2 = int(bounding_box["box"][3])

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        assert "label" in bounding_box
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    return img

def format_detections(detections: List[Bbox]) -> str:
    if not detections:
        return 'No objects detected.'
    return f'Found {len(detections)} objects:\n' + '\n'.join(
        f"- {det['label']}: score {det['score']:.2f}, box {[int(b) for b in det['box']]}"
        for det in detections
    )

def get_dino(max_parallel=1):
    global _dino_factory
    if _dino_factory is None:
        with _dino_lock:
            if _dino_factory is None:
                _dino_factory = DinoObjectDetectionFactory(max_parallel=max_parallel)

    def process_dino(image: Img, objects: List[str], box_threshold=0.2, text_threshold=0.15) -> Tuple[Img, str, List[Bbox]]:
        if not image:
            return None, 'Please upload an image.', []
        if not objects:
            return image, 'Please specify at least one object.', []
        try:
            detections = _dino_factory._run(image, objects, box_threshold, text_threshold)
            drawn_image = plot_bounding_boxes(image.copy(), detections)
            details = format_detections(detections)
            return drawn_image, details, detections
        except Exception as e:
            return image, f'Error: {str(e)}', []

    return process_dino

if __name__ == '__main__':
    detector = get_dino(max_parallel=1)
    # Use detector(image, 'cat, dog', 0.2, 0.1) with an actual image