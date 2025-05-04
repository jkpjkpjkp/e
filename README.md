# Workflow Optimization with aflow.py

This repository contains code for optimizing workflows using aflow.py.

## Issues Fixed

1. **ChatCompletionMessage Handling**: Modified db.py to handle ChatCompletionMessage objects returned by the lmm function.
2. **Simplified Seed File**: Created a simplified seed_modified.py file that doesn't depend on unavailable functions like image_only and groundingdino_operator.

## How to Run

To run aflow.py with the optimized seed file:

```bash
python aflow.py --folder task_main --seed-file data/seed_modified.py --optimization-model claude --inference-model max
```

## Results

When running aflow.py with the modified seed file, we encountered issues with the score calculation due to the ChatCompletionMessage object returned by the lmm function. We fixed this by extracting the content from the ChatCompletionMessage object before passing it to the score function.

The optimization process is still running, but we expect to see improved results with the modified seed file.
