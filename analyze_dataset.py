import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
print("Loading dataset...")
df = pl.read_parquet('dataset.parquet')
print(f"Dataset loaded with {len(df)} rows")

# Function to check if all bounding boxes are enclosed within an area smaller than the entire image
def analyze_bboxes(row):
    annotation = row['annotation']
    width = row['width']
    height = row['height']

    # Calculate total image area
    image_area = width * height

    # Check if we have multiple bounding boxes
    if len(annotation) % 4 != 0:
        print(f"Warning: Annotation length {len(annotation)} is not a multiple of 4")
        return False

    num_boxes = len(annotation) // 4

    # We want at least 7 bounding boxes (numerous)
    if num_boxes < 7:
        return False

    # Find the enclosing bounding box for all annotations
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    for i in range(num_boxes):
        x = annotation[i*4]
        y = annotation[i*4 + 1]
        w = annotation[i*4 + 2]
        h = annotation[i*4 + 3]

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Calculate the area of the enclosing bounding box
    enclosing_area = (max_x - min_x) * (max_y - min_y)

    # Check if the enclosing area is less than the total image area
    return enclosing_area < image_area * 0.9  # Using 90% as threshold

# Apply the analysis to each row
print("Analyzing bounding boxes...")
results = []
for i, row in enumerate(df.iter_rows(named=True)):
    if i % 1000 == 0:
        print(f"Processed {i}/{len(df)} rows")

    if analyze_bboxes(row):
        results.append(row)

# Convert results to a DataFrame
subtask_df = pl.DataFrame(results)
print(f"Found {len(subtask_df)} rows that match the criteria")

# Save the subtask dataset
if len(subtask_df) > 0:
    output_path = 'subtask_dataset.parquet'
    subtask_df.write_parquet(output_path)
    print(f"Subtask dataset saved to {output_path}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Number of unique labels: {subtask_df['label'].n_unique()}")
    print(f"Label distribution:\n{subtask_df.group_by('label').agg(pl.count()).sort('count', descending=True).head(10)}")

    # Calculate and print statistics about the number of bounding boxes
    def count_boxes(annotation):
        return len(annotation) // 4

    subtask_df = subtask_df.with_columns(
        pl.col('annotation').map_elements(count_boxes).alias('num_boxes')
    )

    print(f"\nBounding box statistics:")
    print(f"Min boxes: {subtask_df['num_boxes'].min()}")
    print(f"Max boxes: {subtask_df['num_boxes'].max()}")
    print(f"Mean boxes: {subtask_df['num_boxes'].mean():.2f}")
    print(f"Median boxes: {subtask_df['num_boxes'].median()}")

    # Calculate the ratio of enclosing box area to total image area
    def calculate_area_ratio(row):
        annotation = row['annotation']
        width = row['width']
        height = row['height']
        num_boxes = len(annotation) // 4

        # Find the enclosing bounding box
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0

        for i in range(num_boxes):
            x = annotation[i*4]
            y = annotation[i*4 + 1]
            w = annotation[i*4 + 2]
            h = annotation[i*4 + 3]

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        enclosing_area = (max_x - min_x) * (max_y - min_y)
        image_area = width * height

        return enclosing_area / image_area

    area_ratios = [calculate_area_ratio(row) for row in subtask_df.iter_rows(named=True)]
    print(f"\nEnclosing box to image area ratio statistics:")
    print(f"Min ratio: {min(area_ratios):.4f}")
    print(f"Max ratio: {max(area_ratios):.4f}")
    print(f"Mean ratio: {sum(area_ratios)/len(area_ratios):.4f}")
else:
    print("No matching rows found")
