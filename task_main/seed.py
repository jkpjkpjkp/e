from anode import lmm
from od import grounding_dino

MAX_ITERATIONS = 10

def run(image, question):
    info = []
    for _ in range(MAX_ITERATIONS):
        tool_definition = {
            "type": "function",
            "function": {
                "name": "grounding_dino",
                "description": "Detect objects in an image using Grounding DINO.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "object",
                            "description": "input image"
                        },
                        "objects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of objects to detect in the image"
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": "Threshold for bounding box confidence",
                            "default": 0.2
                        },
                        "text_threshold": {
                            "type": "number",
                            "description": "Threshold for text confidence",
                            "default": 0.15
                        }
                    },
                    "required": [
                        "image",
                        "objects"
                    ]
                }
            }
        }

        response = lmm(image, question, *info, tools=[tool_definition])
        if not response.tool_calls:
            return response.content
        tool_response = globals()[response.tool_calls[0].function.name](**response.tool_calls[0].function.arguments)
        info.append(tool_response)