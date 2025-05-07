from anode import lmm
from od import grounding_dino

MAX_ITERATIONS = 10

def run(image, question):
    # Create a dictionary to map image identifiers to actual images
    image_map = {"image1": image}

    # Create a system prompt to inform the LLM about the image identifiers
    system_prompt = """You are analyzing images. You can refer to images by their identifiers:
- image1: The 1st input image

When using the grounding_dino tool, specify the image using its identifier (e.g., "image1").
"""

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
                            "type": "string",
                            "description": "Image identifier (e.g., 'image1')"
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

        if info:
            info_str = "\n\nPrevious detection results:\n" + "\n".join([str(i) for i in info])
            system_prompt += info_str

        response = lmm(image, question, system_msgs=system_prompt, tools=[tool_definition])
        if not response.tool_calls:
            return response.content

        tool_call = response.tool_calls[0]
        tool_name = tool_call.function.name

        import json
        tool_args = json.loads(tool_call.function.arguments)

        # If the tool is grounding_dino, replace the image identifier with the actual image
        if tool_name == "grounding_dino" and "image" in tool_args and isinstance(tool_args["image"], str):
            image_id = tool_args["image"]
            if image_id in image_map:
                # Replace the image identifier with the actual PIL Image
                tool_args["image"] = image_map[image_id]
            else:
                # If the image identifier is not found, use the first image
                tool_args["image"] = image

        # Call the tool with the processed arguments
        if tool_name == "grounding_dino":
            tool_response = grounding_dino(**tool_args)
        else:
            tool_response = globals()[tool_name](**tool_args)
        info.append(tool_response)