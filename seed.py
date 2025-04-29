from anode import lmm  # a convenient wrapper around lmm api calls, takes in str or Image.Image as args
from PIL import Image

COMPREHENSIVE_VQA_PROMPT = """You are an expert visual question answering system specialized in complex reasoning tasks. Analyze the image and the question meticulously.
1.  **Identify all Rules:** Extract and list all explicit rules, conditions, and parameters mentioned in the question text (e.g., base values, modifiers, multipliers, stopping conditions, selection criteria).
2.  **Extract Visual Data:** Identify and list relevant information from the image for each element involved (e.g., for each plant: pot color, plant type, label, lamp configuration above it, position relative to water).
3.  **Apply Rules Step-by-Step:** For problems involving calculations or simulations (like growth rates), show the detailed calculation for EACH element, applying all relevant rules and visual data identified in the previous steps. Clearly state the formula used for each calculation.
4.  **Determine Final State/Outcome:** Based on the step-by-step application of rules and any stopping conditions or selection criteria mentioned in the question (e.g., which plant reaches a height first, which items are selected), determine the final answer.
5.  **Format Answer:** Present the final answer clearly and concisely, addressing all parts of the original question. If calculations were performed, briefly summarize the key results supporting the answer. Ensure the final answer strictly adheres to any requested format (e.g., comma-separated list, specific value).
Provide only the final answer based *strictly* on the provided image and text. Do not add external knowledge. Show your work clearly as requested above before the final answer line."""

def run(image: Image.Image, question: str) -> str:
    if image.width * image.height > 1500 ** 2:
        image = image.thumbnail((1500, 1500)) # to avoid 'API Payload Too Large'
    ret = lmm(COMPREHENSIVE_VQA_PROMPT, image, question)
    return ret.split('\n')[-1]