from anode import custom
from pydantic import BaseModel, Field
from PIL import Image

class BboxOp(BaseModel):
    thought: str = Field('')
    bbox: tuple[float] = Field('x y x y bbox normalized [0,1]')

def run(image: Image.Image, label: str) -> tuple[float]:
    ret = custom(f'please output the bounding box of "{label}" in the image.', image, dna=BboxOp)['bbox']
    return (
        ret[0] * image.width,
        ret[1] * image.height,
        ret[2] * image.width,
        ret[3] * image.height,
    )
