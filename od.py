import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForCausalLM,
    Owlv2Processor,
    Owlv2ForObjectDetection
)
from typing import List, TypedDict, Tuple
from PIL import Image, ImageDraw, ImageFont
import threading
import requests

class Bbox(TypedDict):
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str

class Dino:
    def __init__(self, max_parallel=3):
        assert torch.cuda.is_available()
        self.device = 'cuda'
        self.gd_processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
        self.semaphore = threading.Semaphore(max_parallel)

    def _run(self, image: Image.Image, texts: List[str], box_threshold=0.2, text_threshold=0.1) -> List[Bbox]:
        """Detect objects in an image using Grounding DINO with concurrency control."""
        with self.semaphore:
            image = image.convert('RGB')
            text = '. '.join(text.strip().lower() for text in texts) + '.'
            print("GD: ", text, image)
            inputs = self.gd_processor(images=image, text=text, return_tensors='pt').to(self.device)
            self.gd_model.to(self.device)

            with torch.no_grad():
                outputs = self.gd_model(**inputs)
                # Move outputs to CPU
                outputs = {k: v.to('cpu') for k, v in outputs.items()}
                # Move input_ids to CPU
                input_ids = inputs['input_ids'].to('cpu')
                results = self.gd_processor.post_process_grounded_object_detection(
                    outputs, input_ids, box_threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[image.size[::-1]]
                )[0]
            assert outputs
            # results = self.gd_processor.post_process_grounded_object_detection(
            #     outputs, inputs['input_ids'], box_threshold=box_threshold,
            #     text_threshold=text_threshold, target_sizes=[image.size[::-1]]
            # )[0]
            return [Bbox(box=box.tolist(), score=score.item(), label=label)
                    for box, score, label in zip(results['boxes'], results['scores'], results['labels'])]

dino = Dino()
def grounding_dino(image: Image.Image, objects: List[str], box_threshold=0.2, text_threshold=0.15) -> Tuple[List[Bbox], Image.Image]:
    """Detect objects in an image using Grounding DINO.
    
    Args:
        image: Input image.
        objects: List of objects to detect in the image.
        box_threshold: Threshold for bounding box confidence.
        text_threshold: Threshold for text confidence.

    Returns:
        A tuple (
            the list of bbox,
            image with bbox drawn,
        )
    """
    assert isinstance(image, Image.Image)
    assert isinstance(objects, list)
    assert all(isinstance(x, str) for x in objects)
    assert 0 <= box_threshold <= 1
    assert 0 <= text_threshold <= 1
    image = image.convert('RGB')
    detections = dino._run(image, objects, box_threshold, text_threshold)
    drawn_image = plot_bounding_boxes(image.copy(), detections)
    return detections, drawn_image

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

    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    font = ImageFont.load_default(size=14)


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

class Owl:
    def __init__(self, max_parallel=3):
        self.device = 'cuda'
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(self.device).half()
        self.semaphore = threading.Semaphore(max_parallel)

    def _run(self, image: Image.Image, texts: List[str], threshold=0.1) -> List[Bbox]:
        if not texts or not image:
            raise ValueError('Valid image and at least one object description required')
        with self.semaphore:
            image = image.convert('RGB')
            text_queries = [["a photo of " + text for text in texts]]
            inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                label_name = texts[label]
                detections.append(Bbox(box=box.tolist(), score=score.item(), label=label_name))
            return detections

owl = Owl()

def owl_v2(image: Image.Image, objects: List[str], threshold=0.1) -> Tuple[List[Bbox], Image.Image]:
    """Detect objects in an image using OWLv2.
    
    Args:
        image: Input image.
        objects: List of objects to detect in the image.
        threshold: Confidence score threshold.

    Returns:
        A tuple (
            the list of bbox,
            image with bbox drawn,
        )
    """
    assert isinstance(image, Image.Image)
    assert isinstance(objects, list)
    assert all(isinstance(x, str) for x in objects), objects
    assert 0 <= threshold <= 1, threshold
    image = image.convert('RGB')
    detections = owl._run(image, objects, threshold)
    drawn_image = plot_bounding_boxes(image.copy(), detections)
    return detections, drawn_image

# class Florence:
#     def __init__(self, max_parallel: int = 1):
#         self.device = 'cuda'
#         self.torch_dtype = torch.float16
#         from transformers import AutoModelForCausalLM
#         # self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "microsoft/Florence-2-large",
#             torch_dtype=self.torch_dtype,
#             trust_remote_code=True
#         ).to(self.device)
#         self.processor = AutoProcessor.from_pretrained(
#             "microsoft/Florence-2-large",
#             trust_remote_code=True
#         )
#         self.semaphore = threading.Semaphore(max_parallel)

#     def detect_objects(self, image: Image.Image, target_labels: List[str]) -> List[Bbox]:
#         if not target_labels or not image:
#             raise ValueError("A valid image and at least one target label are required.")

#         with self.semaphore:
#             image = image.convert('RGB')
#             prompt = "<OD>"

#             inputs = self.processor(
#                 text=prompt,
#                 images=image,
#                 return_tensors="pt"
#             ).to(self.device, self.torch_dtype)

#             generated_ids = self.model.generate(
#                 input_ids=inputs["input_ids"],
#                 pixel_values=inputs["pixel_values"],
#                 max_new_tokens=4096,
#                 num_beams=3,
#                 do_sample=False
#             )

#             generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#             parsed_result = self.processor.post_process_generation(
#                 generated_text,
#                 task="<OD>",
#                 image_size=(image.width, image.height)
#             )

#             boxes = parsed_result.get('boxes', [])
#             labels = parsed_result.get('labels', [])
#             detections = []

#             for box, label in zip(boxes, labels):
#                 if label in target_labels:
#                     detections.append(Bbox(box=box, score=1.0, label=label))

#             return detections

# florence = Florence()

# def florence_v2(image: Image.Image, objects: List[str]) -> Tuple[List[Bbox], Image.Image]:
#     """Detect objects in an image using Florence V2.

#     Args:
#         image: Input image.
#         objects: List of objects to detect in the image.

#     Returns:
#         A tuple (
#             the list of bbox,
#             image with bbox drawn,
#         )
#     """
#     image = image.convert('RGB')
#     detections = florence._run(image, objects)
#     drawn_image = plot_bounding_boxes(image.copy(), detections)
#     return detections, drawn_image

# def _florence_demo():
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
#     processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

#     prompt = "<OD>"

#     url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#     image = Image.open(requests.get(url, stream=True).raw)

#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         max_new_tokens=4096,
#         num_beams=3,
#         do_sample=False
#     )
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

#     parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

#     print(parsed_answer)


def _owl_demo():

    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


tools = [
    {
        "name": "grounding_dino",
        "description": "Detect objects in an image using Grounding DINO.",
        "callable": grounding_dino,
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "Image.Image",
                    "description": "input image"
                },
                "objects": {
                    "type": "List[str]",
                    "description": "List of objects to detect in the image"
                },
                "box_threshold": {
                    "type": "float",
                    "description": "Threshold for bounding box confidence",
                    "default": 0.2
                },
                "text_threshold": {
                    "type": "float",
                    "description": "Threshold for text confidence",
                    "default": 0.15
                },
            },
            "required": [
                "image",
                "objects",
            ],
            "additionalProperties": False,
            "returns": {
                "bbox": {
                    "type": "List[Bbox]",
                    "description": "List of bounding boxes for detected objects"
                },
                "image": {
                    "type": "Image.Image",
                    "description": "Image with bounding boxes drawn"
                },
            },
        }
    },
    {
        "name": "florence",
        "description": "Detect objects in an image using Florence V2.",
    },
]

for tool in tools:
    tool["type"] = "function"


operators = [grounding_dino, owl_v2]

def test_operators():
    image = Image.open('/hy-tmp/count/train/010be05b8bd90c7d.jpg')
    objects = ['person', 'dog', 'cat', 'car']
    print(grounding_dino(image, objects))
    print(owl_v2(image, objects))

if __name__ == '__main__':
    test_operators()