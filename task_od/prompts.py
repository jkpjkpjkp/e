
WORKFLOW_OPTIMIZE_PROMPT = """We are optimizing a function `run` that accurately detects objects. 

you can use 2 leading object grounding models.
your code should return a List[Bbox].
`
class Bbox(TypedDict):
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str
`
your code should not fallback to dummy return in any situation. raise if any errors occur. 

Output the modified code under the same setting and function name (run).

Here is a graph that performed excellently in a previous iteration for VQA. You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <score>{score}</score>
    <graph>{graph}</graph>
    <operator_description>{operator_description}</operator_description>
</sample>
"""

WORKFLOW_OPTIMIZE_GUIDANCE="""
First, analyze the trace, brainstorm, and propose optimization ideas. **Only one detail should be modified**, and **no more than 5 lines of code should be changed**—extensive modifications are strictly prohibited to maintain project focus! Simplifying code by removing unnecessary steps is often highly effective. When adding new functionalities to the graph, ensure necessary libraries or modules are imported, including importing operators from `op`.
"""

OPERATORS = '''
def grounding_dino(image: Image.Image, objects: List[str], box_threshold=0.2, text_threshold=0.15) -> Tuple[List[Bbox], Image.Image]:
    """Detect objects in an image using Grounding DINO.
    
    Args:
        image: Input image.
        objects: List of objects to detect in the image.
        box_threshold: Threshold for bounding box confidence.
        text_threshold: Threshold for text confidence.

    Returns:
        A tuple (
            the list of bbox,
            image with bbox drawn,
        )
    """
    ...


def owl_v2(image: Image.Image, objects: List[str], threshold=0.1) -> Tuple[List[Bbox], Image.Image]:
    """Detect objects in an image using OWLv2.
    
    Args:
        image: Input image.
        objects: List of objects to detect in the image.
        threshold: Confidence score threshold.

    Returns:
        A tuple (
            the list of bbox,
            image with bbox drawn,
        )
    """
    ...

'''
def optimize(graph):
    raise