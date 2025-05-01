from PIL import Image
import os

image_cache_dir = '/data/image_cache'
def find_and_convert(tp, f):
    def convert(obj, seen=None):
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return obj

        seen.add(obj_id)

        # Process the object
        if isinstance(obj, tp):
            return f(obj)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return {k: convert(v, seen) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x, seen) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(x, seen) for x in obj)
        else:
            return obj

    return convert

def img_to_str(image: Image.Image):
    assert isinstance(image, Image.Image)
    image_hash = hash(image.tobytes())
    image_path = os.path.join(image_cache_dir, f'{image_hash}.png')
    if not os.path.exists(image_path):
        image.save(image_path)
    return f"<__imimaimage>{image_path}</__imimaimage>"

def unit_str_to_img(s: str):
    if s.startswith('<__imimaimage>') and s.endswith('</__imimaimage>'):
        return Image.open(s[len('<__imimaimage>'): -len('</__imimaimage>')])
    else:
        return s

def str_to_img(s: str):
    assert isinstance(s, str)
    # Simple case: single image
    if s.startswith('<__imimaimage>') and s.endswith('</__imimaimage>'):
        return Image.open(s[len('<__imimaimage>'): -len('</__imimaimage>')])

    # Check if there are any image markers
    if '<__imimaimage>' not in s:
        return s

    # Complex case: text with embedded images
    parts = []
    segments = s.split('<__imimaimage>')

    # Add the text before the first image marker
    if segments[0]:
        parts.append(segments[0])

    # Process each image segment
    for segment in segments[1:]:
        if '</__imimaimage>' in segment:
            # Split at the end marker
            img_path, remaining = segment.split('</__imimaimage>', 1)
            # Add the image
            parts.append(Image.open(img_path))
            # Add any remaining text
            if remaining:
                parts.append(remaining)
        else:
            # If no end marker, treat as text
            parts.append(segment)

    # Return as tuple for JSON serialization compatibility
    return tuple(parts)
