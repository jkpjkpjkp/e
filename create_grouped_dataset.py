import polars as pl

# Read the original dataset
df = pl.read_parquet('dataset_grouped.parquet')

# Group by image_path to get all labels and annotations for each image
image_groups = df.group_by('image_path').agg(
    pl.col('batch').first().alias('batch'),
    pl.col('image_id').first().alias('image_id'),
    pl.col('width').first().alias('width'),
    pl.col('height').first().alias('height'),
    pl.col('label').alias('labels'),
    pl.col('annotations').alias('annotations_by_label')
)

# Process the data to create the new format
result_rows = []
for row in image_groups.iter_rows(named=True):
    image_path = row['image_path']
    batch = row['batch']
    image_id = row['image_id']
    width = row['width']
    height = row['height']
    labels = row['labels']
    annotations_by_label = row['annotations_by_label']

    # Combine all annotations with their label indices
    all_annotations = []
    for i, label_annotations in enumerate(annotations_by_label):
        for bbox in label_annotations:
            # Add the label index to each bounding box as a float
            # bbox is [x, y, w, h], we add the label index as the 5th element
            all_annotations.append(bbox + [float(i)])

    result_rows.append({
        'image_path': image_path,
        'batch': batch,
        'image_id': image_id,
        'width': width,
        'height': height,
        'labels': labels,
        'all_annotations': all_annotations
    })

# Create the new dataframe with strict=False to allow mixed types
result_df = pl.DataFrame(result_rows, strict=False)

# Save the new dataset
result_df.write_parquet('dataset_all_grouped.parquet')

print("Created new dataset with grouped bounding boxes")
print(f"Original dataset shape: {df.shape}")
print(f"New dataset shape: {result_df.shape}")

# Print a sample to verify
print("\nSample from new dataset:")
sample = result_df.head(1)
print(sample.select(['image_path', 'labels']))
print("First row annotations:")
print(sample['all_annotations'][0])
