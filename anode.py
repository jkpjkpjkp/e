import re
from typing import Any, Dict, Type
from pydantic import BaseModel, Field
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from loguru import logger
from typing import get_args


def to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}'

class LLM:
    def __init__(self, model) -> None:
        self.model = model
        self.client = OpenAI(
            api_key='sk-local',
            base_url='http://localhost:7912'
        )

    def aask(self, prompt, system_msgs=''):
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
                    content.append({"type": "image_url", "image_url": {"url": to_base64(stuff)}})
            messages.append({"role": "user", "content": content})

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        ).choices[0].message.content

def lmm(*args, **kwargs):
    ret = LLM(model='qwen-vl-max-latest').aask(prompt=args, **kwargs)
    return ret


class ActionNode:
    def __init__(self, pydantic_model: Type[BaseModel]):
        self.pydantic_model = pydantic_model

    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand the xml format.
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

    def fill(
        self,
        *,
        context,
        llm,
    ):
        self.llm = llm
        self.context = context
        context = self.xml_compile(context=self.context)
        result = self.xml_fill(context)
        result = self.before_validator(result)
        return self.pydantic_model(**result)


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

def custom(*args, dna=GenerateOp):
    for x in args:
        assert x
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]
    return ActionNode(dna).fill(context=args, llm=LLM(model='claude-3-7-sonnet-20250219'))

def test_custom():
    print(custom('hi! '))
