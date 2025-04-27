from action_node import custom
from pydantic import BaseModel, Field

class BboxOp(BaseModel):
    bbox: tuple[int, int, int, int] = Field('x y x y bbox normalized to [0-1000]')

def run(image, label) -> tuple[float]:
    ret = custom(f'please output the bounding box of "{label}" in the image.', image, dna=BboxOp)['bbox']
    return (
        ret[0] * image.width / 1000,
        ret[1] * image.height / 1000,
        ret[2] * image.width / 1000,
        ret[3] * image.height / 1000,
    )
