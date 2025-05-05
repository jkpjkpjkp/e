from anode import lmm, parse
from od import grounding_dino

MAX_ITERATIONS = 10

def run(image, question):
    info = []
    for _ in range(MAX_ITERATIONS):
        response = lmm(image, question, *info, tools = [{
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
    }])
        if not response.tool_calls:
            return response.content
        tool_response = globals()[response.tool_calls[0].function.name](**response.tool_calls[0].function.arguments)
        info.append(tool_response)