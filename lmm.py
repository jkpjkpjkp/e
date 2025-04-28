import re
import typing
from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel, Field
from openai import OpenAI
import base64
from io import BytesIO
import asyncio
from PIL import Image
from pydantic import BeforeValidator
from typing_extensions import Annotated
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
    return LLM().aask(prompt=args, **kwargs)
