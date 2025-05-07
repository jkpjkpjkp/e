import re
import json
import inspect
from typing import Any, Dict, Type, List, Optional, Callable
from pydantic import BaseModel, Field
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from loguru import logger
from typing import get_args
from util import *
# Global variables for model names
inference_model = None
optimization_model = None

# Setter functions
def set_inference_model(model_name):
    global inference_model
    inference_model = model_name

def set_optimization_model(model_name):
    global optimization_model
    optimization_model = model_name

# Export the parse function for use in other modules
__all__ = ['LLM', 'lmm', 'custom', 'ActionNode', 'parse',
           'set_inference_model', 'set_optimization_model']



def to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}'

def make_json_serializable(obj):
    """
    Recursively convert non-serializable objects to string representations.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        if 'function' in obj and isinstance(obj['function'], dict) and 'callable' in obj['function']:
            result = {}
            for k, v in obj.items():
                if k == 'function':
                    function_dict = {}
                    for fn_k, fn_v in obj['function'].items():
                        if fn_k == 'callable' and callable(fn_v):
                            logger.warning(f"Callable function {fn_v} is not serializable. dropped.")
                            continue
                        else:
                            function_dict[fn_k] = make_json_serializable(fn_v)
                    result[k] = function_dict
                else:
                    result[k] = make_json_serializable(v)
            return result
        else:
            return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif callable(obj):
        return f"<function {obj.__name__}>"
    elif hasattr(obj, "__dict__"):
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    else:
        return str(obj)

class LLM:
    def __init__(self, model) -> None:
        self.model = model
        self.client = OpenAI(
            api_key='sk-local',
            base_url='http://localhost:7912'
        )

    def aask(self, prompt, system_msgs='', tools=None):
        messages = [
            {"role": "system", "content": system_msgs or "You are a helpful assistant. "},
        ]
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        else:
            assert isinstance(prompt, tuple) or isinstance(prompt, list)
            content = []
            for stuff in prompt:
                if isinstance(stuff, str):
                    content.append({"type": "text", "text": stuff})
                elif stuff is None:
                    pass
                else:
                    assert isinstance(stuff, Image.Image), prompt
                    stuff.thumbnail((1512, 1512))
                    assert min(stuff.height, stuff.width) > 11
                    content.append({"type": "image_url", "image_url": {"url": to_base64(stuff)}})
            messages.append({"role": "user", "content": content})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        if tools:
            kwargs["tools"] = make_json_serializable(tools)

        kwargs = make_json_serializable(kwargs)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

def lmm(*args, **kwargs):
    global inference_model

    if inference_model is None:
        raise ValueError("inference_model is not set. Please call set_inference_model() before using lmm().")

    processed_args = [x if isinstance(x, str) else x.copy().convert('RGB') for x in args]

    for item in processed_args:
        if isinstance(item, Image.Image):
            item.thumbnail((1512, 1512))
            assert min(item.width, item.height) > 11, "Image dimension too small"

    assert any(isinstance(item, Image.Image) for item in processed_args), "No images provided"

    for item in processed_args:
        if isinstance(item, Image.Image):
            assert item.width * item.height <= 1512 ** 2, f"Image size {item.size} exceeds 1500x1500 pixels"

    deduplicated_args = []
    for item in processed_args:
        if isinstance(item, str):
            deduplicated_args.append(item)
        else:
            if item not in deduplicated_args:
                deduplicated_args.append(item)
            else:
                deduplicated_args.append('<image>')

    client = OpenAI(
        api_key='sk-local',
        base_url='http://localhost:7912'
    )

    messages = [
        {"role": "system", "content": kwargs.get('system_msgs', '') or "You are a helpful assistant. "},
    ]

    content = []
    for stuff in deduplicated_args:
        if isinstance(stuff, str):
            content.append({"type": "text", "text": stuff})
        elif stuff is None:
            pass
        else:
            assert isinstance(stuff, Image.Image)
            stuff.thumbnail((1512, 1512))
            assert min(stuff.height, stuff.width) > 11
            content.append({"type": "image_url", "image_url": {"url": to_base64(stuff)}})

    messages.append({"role": "user", "content": content})

    api_kwargs = {
        "model": inference_model or 'claude-3-7-sonnet-20250219',
        "messages": messages,
        "temperature": 0,
    }

    if 'tools' in kwargs:
        print(f"Tools provided: {len(kwargs['tools'])} tool(s)")
        api_kwargs["tools"] = make_json_serializable(kwargs['tools'])
        print(f"Tools serialized: {len(api_kwargs['tools'])} tool(s)")

    # Avoid printing the entire API request which may contain base64 image data
    print(f"API request prepared with model: {api_kwargs['model']}")
    api_kwargs = make_json_serializable(api_kwargs)
    print("API request serialized successfully")

    try:
        import json
        json.dumps(api_kwargs)  # Just check if it can be serialized, don't store the result
        print("Successfully serialized to JSON")
    except Exception as e:
        print(f"Error serializing to JSON: {e}")
        # Try to identify the problematic part without printing the values
        for key, value in api_kwargs.items():
            try:
                json.dumps({key: value})
            except Exception as e:
                print(f"Problem with key {key}: {e}")
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        try:
                            json.dumps(item)
                        except Exception as e:
                            print(f"Problem with item {i} in {key}: {e}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        try:
                            json.dumps({k: v})
                        except Exception as e:
                            print(f"Problem with key {k} in {key}: {e}")
        raise

    response = client.chat.completions.create(**api_kwargs)

    return response.choices[0].message

class ActionNode:
    def __init__(self, pydantic_model: Type[BaseModel]):
        self.pydantic_model = pydantic_model

    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand intended output xml format.
        """
        # Construct the example using the field names and their types
        examples = []
        for field_name, field_info in self.pydantic_model.model_fields.items():
            examples.append(f"<{field_name}> # type: {field_info.annotation.__name__}{', ' + field_info.description if field_info.description else ''}{', default=' + str(field_info.default) if str(field_info.default) and str(field_info.default) != 'PydanticUndefined' else ''}</{field_name}>")

        # Join all examples into a single string
        example_str = "\n".join(examples)
        # Add the example to the context
        xml_hint = f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""
        if isinstance(context, str):
            context += xml_hint
        else:
            context = (*context, xml_hint)
        return context

    def xml_fill(self, context: str | tuple) -> Dict[str, Any]:
        extracted_data: Dict[str, Any] = {}
        content = self.llm.aask(context)
        for field_name, field_info in self.pydantic_model.model_fields.items():
            pattern = rf"<{field_name}>((?:(?!<{field_name}>).)*?)</{field_name}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                field_type = field_info.annotation
                if field_type is str:
                    pattern = r"```python(.*?)```"
                    match = re.search(pattern, raw_value, re.DOTALL)
                    if match:
                        raw_value = '\n'.join(match.groups())
                    extracted_data[field_name] = raw_value
                elif field_type is int:
                    try:
                        extracted_data[field_name] = int(raw_value)
                    except ValueError:
                        extracted_data[field_name] = 0
                elif field_type is bool:
                    extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
                elif field_type is list:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], list):
                            raise ValueError
                    except Exception as e:
                        logger.warning(f"Failed to parse {raw_value} as list: {e}")
                        extracted_data[field_name] = []
                elif field_type is dict:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], dict):
                            raise ValueError
                    except Exception as e:
                        logger.warning(f"Failed to parse {raw_value} as dict: {e}")
                        extracted_data[field_name] = {}
                else:
                    extracted_data[field_name] = raw_value
            else:
                pattern = rf"^(.*?)</{field_name}>"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    raw_value = match.group(1).strip()
                    extracted_data[field_name] = raw_value
        return extracted_data

    def before_validator(self, result):
        types = self.pydantic_model.model_fields
        for k, v in result.items():
            field_info = types[k]
            if hasattr(field_info.annotation, "__origin__") and field_info.annotation.__origin__ is tuple:
                inner_type = get_args(field_info.annotation)[0]
                if isinstance(v, str):
                    if v.startswith('['):
                        v = v[1:-1]
                    elif v.startswith('('):
                        v = v[1:-1]
                    v = v.replace(',', ' ').strip().split()
                    v = (x for x in v if x)
                result[k] = tuple(inner_type(x) for x in v)
        return result

    def fill(self, *, context, llm):
        self.llm = llm
        self.context = context
        context = self.xml_compile(context=self.context)
        result = self.xml_fill(context)
        result = self.before_validator(result)
        return self.pydantic_model(**result)


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

def custom(*args, dna=GenerateOp):
    global optimization_model

    if optimization_model is None:
        raise ValueError("optimization_model is not set. Please call set_optimization_model() before using custom().")

    for x in args:
        assert x
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]

    # Convert string arguments to images if they contain image markers
    processed_args = []
    for arg in args:
        if isinstance(arg, str):
            oo = str_to_img(arg)
            if isinstance(oo, (str, Image.Image)):
                processed_args.append(oo)
            else:
                # Our updated str_to_img returns a tuple
                processed_args.extend(oo)
        else:
            # Just append non-string arguments (like images) directly
            processed_args.append(arg)

    return ActionNode(dna).fill(context=processed_args, llm=LLM(model=optimization_model))

def test_custom():
    print(custom('hi! '))
