import re
import typing
from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel, Field, create_model
from openai import OpenAI
import base64
from io import BytesIO
import asyncio
from PIL import Image
from pydantic import BeforeValidator
from typing_extensions import Annotated
from loguru import logger


def to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}'

class LLM:
    def __init__(self, model='claude-3-7-sonnet-20250219') -> None:
        self.model = model
        self.client = OpenAI(
            api_key='sk-local',
            base_url='http://localhost:7912'
        )

    def aask(self, prompt, system_msgs=''):

        print(prompt)
        raise
        messages = [
            {"role": "system", "content": system_msgs or "You are a helpful assistant."},
        ]
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        else:
            assert isinstance(prompt, tuple) or isinstance(prompt, list)
            content = []
            for stuff in prompt:
                if isinstance(stuff, str):
                    content.append({"type": "text", "text": stuff})
                else:
                    assert isinstance(stuff, Image.Image)
                    content.append({"type": "image_url", "image_url": {"url": to_base64(stuff)}})
            messages.append({"role": "user", "content": content})
        
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        ).choices[0].message.content

class ActionNode:
    def __init__(
        self,
        key: str,
        expected_type: Type,
        instruction: str,
        example: Any,
        content: str = "",
        children: dict[str, "ActionNode"] = None,
        schema: str = "",
    ):
        self.key = key
        self.expected_type = expected_type
        self.instruction = instruction
        self.example = example
        self.content = content
        self.children = children if children is not None else {}
        self.schema = schema

    def add_child(self, node: "ActionNode"):
        self.children[node.key] = node

    def get_mapping(self, mode="children", exclude=None) -> Dict[str, Tuple[Type, Any]]:
        exclude = exclude or []
        if mode == "children" or (mode == "auto" and self.children):
            mapping = {}
            for key, child in self.children.items():
                if key in exclude:
                    continue
                mapping[key] = (child.expected_type, Field(default=child.example, description=child.instruction))
            return mapping
        return {} if exclude and self.key in exclude else {self.key: (self.expected_type, ...)}

    @classmethod
    def create_model_class(cls, class_name: str, mapping: Dict[str, Tuple[Type, Any]]):
        new_fields = {}
        for field_name, (field_type, field_info) in mapping.items():
            new_fields[field_name] = (field_type, field_info)
        return create_model(class_name, **new_fields)

    def create_class(self, mode="auto", class_name: str = None, exclude=None):
        class_name = class_name if class_name else f"{self.key}_AN"
        mapping = self.get_mapping(mode=mode, exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def get_field_names(self):
        model_class = self.create_class()
        return model_class.model_fields.keys()

    def get_field_types(self):
        model_class = self.create_class()
        return {field_name: field.annotation for field_name, field in model_class.model_fields.items()}

    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand the xml format.
        """
        field_names = self.get_field_names()
        field_types = self.get_field_types()
        # Construct the example using the field names and their types
        examples = []
        for field_name in field_names:
            field_type = field_types.get(field_name)
            examples.append(f"<{field_name}> # type: {field_type}</{field_name}>")

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
        field_names = self.get_field_names()
        field_types = self.get_field_types()
        extracted_data: Dict[str, Any] = {}
        content = self.llm.aask(context)
        for field_name in field_names:
            pattern = rf"<{field_name}>((?:(?!<{field_name}>).)*?)</{field_name}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                field_type = field_types.get(field_name)
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
        self.instruct_content = self.dantic(**result)
        return self

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], key: str = None):
        key = key or model.__name__
        root_node = cls(key=key, expected_type=Type[model], instruction="", example="")
        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description or ""
            default = field_info.default
            if not isinstance(field_type, typing._GenericAlias) and issubclass(field_type, BaseModel):
                child_node = cls.from_pydantic(field_type, key=field_name)
            else:
                child_node = cls(key=field_name, expected_type=field_type, instruction=description, example=default)
            root_node.add_child(child_node)
        root_node.dantic = model
        return root_node



class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

def parse_bbox_string(v: str) -> Tuple[int, int, int, int]:
    if v.startswith('['):
        v = v[1:-1]
    elif v.startswith('('):
        v = v[1:-1]
    v = v.replace(',', ' ')
    try:
        numbers = v.split()
        if len(numbers) != 4:
            raise ValueError("Expected four numbers in bbox string")
        return tuple(int(num) for num in numbers)
    except (ValueError, TypeError):
        raise ValueError("Invalid bbox format; expected 'num num num num'")

BBoxType = Annotated[Tuple[int, int, int, int], BeforeValidator(parse_bbox_string)]

class CropOp(BaseModel):
    thought: str = Field(default="", description="Thoughts on what crop may be sufficient.")
    bbox: BBoxType = Field(..., description="a crop containing all relevant information, in x y x y format, idx from 0 to 1000. ")


def custom(*args, dna=GenerateOp):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]
    return ActionNode.from_pydantic(dna).fill(context=args, llm=LLM())

def test_custom():
    print(custom('hi! '))
