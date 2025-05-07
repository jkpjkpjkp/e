from anode import lmm
from od import grounding_dino

MAX_ITERATIONS = 10

def run(image, question):
    # Create a dictionary to map image identifiers to actual images
    image_map = {"image1": image}

    # Create a system prompt to inform the LLM about the image identifiers
    system_prompt = """You are analyzing images to answer questions accurately. You can refer to images by their identifiers:
- image1: The 1st input image

IMPORTANT INSTRUCTIONS:
1. For counting questions, use the grounding_dino tool to detect and count objects.
2. Always provide a clear, concise numerical answer.
3. For questions asking "how many", make sure to use object detection to count accurately.
4. Your final answer should be a single number or a very short phrase.
5. Always end your response with "The answer is: X" where X is your final answer.

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
            # Format detection results without printing the image object
            formatted_info = []
            for i in info:
                assert isinstance(i, tuple) and len(i) == 2  # detections and image
                detections, _img = i
                formatted_info.append(f"Detections: {detections}")

            info_str = "\n\nPrevious detection results:\n" + "\n".join(formatted_info)
            system_prompt += info_str

        response = lmm(image, question, system_msgs=system_prompt, tools=[tool_definition])
        if not response.tool_calls:
            # Extract the final answer if it follows our format
            import re
            answer_match = re.search(r"The answer is:\s*(\d+|[\w\s\-\.]+)(?:\.|$)", response.content)
            if answer_match:
                return answer_match.group(1).strip()
            else:
                # If no formatted answer found, return the full content
                return response.content

        tool_call = response.tool_calls[0]
        tool_name = tool_call.function.name

        import json
        # Ensure tool_args is a dictionary, not a string
        if isinstance(tool_call.function.arguments, str):
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Arguments: {tool_call.function.arguments}")
                # If JSON parsing fails, create a basic dictionary with the image
                tool_args = {"image": image, "objects": ["person", "car"]}
        else:
            tool_args = tool_call.function.arguments

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