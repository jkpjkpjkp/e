import polars as pl
import numpy as np
from pathlib import Path

# Load the dataset
print("Loading dataset...")
df = pl.read_parquet('dataset.parquet')
print(f"Dataset loaded with {len(df)} rows")

# First, let's understand the structure
print("\nDataset structure:")
print(f"Number of unique image_path values: {df['image_path'].n_unique()}")
print(f"Number of unique image_id values: {df['image_id'].n_unique()}")
print(f"Number of unique (image_path, label) pairs: {df.group_by(['image_path', 'label']).agg(pl.count()).shape[0]}")

# Group by image_path and label to get all annotations for each image-label pair
print("\nGrouping annotations by image-label pairs...")
grouped = df.group_by(['image_path', 'label', 'width', 'height', 'image_id'])

# Aggregate the annotations
def concat_annotations(annotations):
    all_annotations = []
    for annotation in annotations:
        all_annotations.extend(annotation)
    return all_annotations

# Create a new dataframe with concatenated annotations
agg_df = grouped.agg(
    pl.col('annotation').apply(concat_annotations).alias('all_annotations'),
    pl.col('annotation_id').count().alias('annotation_count')
)

print(f"Created {len(agg_df)} image-label pairs")

# Function to analyze the bounding boxes for each image-label pair
def analyze_bboxes(row):
    all_annotations = row['all_annotations']
    width = row['width']
    height = row['height']
    annotation_count = row['annotation_count']

    # Calculate total image area
    image_area = width * height

    # Check if we have enough annotations (at least 7)
    if annotation_count < 7:
        return False

    # Check if annotations are in the correct format (multiples of 4)
    if len(all_annotations) % 4 != 0:
        print(f"Warning: Annotation length {len(all_annotations)} is not a multiple of 4")
        return False

    num_boxes = len(all_annotations) // 4

    # Find the enclosing bounding box for all annotations
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    for i in range(num_boxes):
        x = all_annotations[i*4]
        y = all_annotations[i*4 + 1]
        w = all_annotations[i*4 + 2]
        h = all_annotations[i*4 + 3]

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
for i, row in enumerate(agg_df.iter_rows(named=True)):
    if i % 1000 == 0:
        print(f"Processed {i}/{len(agg_df)} rows")

    if analyze_bboxes(row):
        results.append(row)

# Convert results to a DataFrame
subtask_df = pl.DataFrame(results)
print(f"Found {len(subtask_df)} image-label pairs that match the criteria")

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
    def count_boxes(annotations):
        return len(annotations) // 4

    subtask_df = subtask_df.with_columns(
        pl.col('all_annotations').map_elements(count_boxes).alias('num_boxes')
    )

    print(f"\nBounding box statistics:")
    print(f"Min boxes: {subtask_df['num_boxes'].min()}")
    print(f"Max boxes: {subtask_df['num_boxes'].max()}")
    print(f"Mean boxes: {subtask_df['num_boxes'].mean():.2f}")
    print(f"Median boxes: {subtask_df['num_boxes'].median()}")

    # Calculate the ratio of enclosing box area to total image area
    def calculate_area_ratio(row):
        all_annotations = row['all_annotations']
        width = row['width']
        height = row['height']
        num_boxes = len(all_annotations) // 4

        # Find the enclosing bounding box
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0

        for i in range(num_boxes):
            x = all_annotations[i*4]
            y = all_annotations[i*4 + 1]
            w = all_annotations[i*4 + 2]
            h = all_annotations[i*4 + 3]

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

    # Save a sample of 5 image-label pairs for visualization
    sample_df = subtask_df.sample(n=min(5, len(subtask_df)), seed=42)
    sample_path = 'sample_subtask.parquet'
    sample_df.write_parquet(sample_path)
    print(f"\nSaved a sample of {len(sample_df)} image-label pairs to {sample_path}")
else:
    print("No matching rows found")
