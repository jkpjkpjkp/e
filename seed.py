from anode import custom
from pydantic import BaseModel, Field
from PIL import Image

class BboxOp(BaseModel):
    thought: str = Field('')
    bbox: tuple[float, float, float, float] = Field(..., description='x y x y bbox normalized to [0,1]')

def run(image: Image.Image, label: str) -> tuple[float, float, float, float]:
    assert isinstance(image, Image.Image)
    ret = custom(f'please output the bounding box of "{label}" in the image.', image, dna=BboxOp).bbox
    return (
        ret[0] * image.width,
        ret[1] * image.height,
        ret[2] * image.width,
        ret[3] * image.height,
    )
