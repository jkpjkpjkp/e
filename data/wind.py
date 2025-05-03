from PIL import Image
from op import lmm

def run(image, question) -> tuple[int, int, int, int]:
    return (0, 0, image.width, image.height)
