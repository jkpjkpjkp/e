# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="microsoft/Florence-2-large", trust_remote_code=True)