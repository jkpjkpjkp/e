import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import numpy as np

# Load the dataset
df = pl.read_parquet('dataset_grouped.parquet')

# Get the first few examples
examples = df.head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, ax in enumerate(axes):
    # Get image path and annotations
    img_path = examples['image_path'][i]
    annotations = examples['annotations'][i]
    
    # Since we can't access the actual images, create a blank image with the right dimensions
    width = examples['width'][i]
    height = examples['height'][i]
    blank_img = np.ones((height, width, 3))
    
    ax.imshow(blank_img)
    
    # Draw bounding boxes
    for ann in annotations:
        x, y, w, h = ann
        
        # Draw box assuming x,y is top-left corner (red)
        rect_corner = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_corner)
        
        # Draw box assuming x,y is center (blue)
        rect_center = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=2, edgecolor='b', facecolor='none', linestyle='--')
        ax.add_patch(rect_center)
    
    # Add annotations to explain
    ax.text(10, 20, f"Red: x,y as top-left", color='r')
    ax.text(10, 40, f"Blue dashed: x,y as center", color='b')
    
    # Add coordinates info
    for j, ann in enumerate(annotations):
        x, y, w, h = ann
        ax.text(10, 60 + j*20, f"Box {j+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}", fontsize=8)
    
    ax.set_title(f"Image {i+1}: {width}x{height}")

plt.tight_layout()
plt.savefig('bbox_visualization.png')
print("Visualization saved to bbox_visualization.png")

# Also print the coordinates and image dimensions for reference
for i in range(3):
    ann = examples['annotations'][i][0]
    w = examples['width'][i]
    h = examples['height'][i]
    x, y, width, height = ann
    print(f'Example {i+1}:')
    print(f'Annotation: x={x}, y={y}, width={width}, height={height}')
    print(f'Image dimensions: width={w}, height={h}')
    print(f'Right edge if corner: {x + width}, Bottom edge if corner: {y + height}')
    print(f'Left edge if center: {x - width/2}, Top edge if center: {y - height/2}')
    print(f'Right edge if center: {x + width/2}, Bottom edge if center: {y + height/2}')
    print()
