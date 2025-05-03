from PIL import Image
import os
import random
from florence import G_Dino, pipeline_detector

def test_pipeline_vs_class():
    print("Comparing G_Dino class vs pipeline implementation...")
    
    # Get a list of image files from the training directory
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    # Select a random image for testing
    image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, image_file)
    print(f"\nTesting with image: {image_path}")
    
    # Load the image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Create G_Dino instance
    g_dino = G_Dino()
    
    # Common objects to detect
    objects = ["person", "car", "dog", "cat", "chair", "table"]
    
    # Test with different thresholds
    threshold = 0.2
    print(f"Using threshold: {threshold}")
    
    # Test G_Dino class
    try:
        print("\nUsing G_Dino class:")
        class_detections = g_dino.detect(image, objects, box_threshold=threshold)
        
        print(f"Detected {len(class_detections)} objects:")
        for det in class_detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")
        
        if class_detections:
            result_image = g_dino.draw_boxes(image.copy(), class_detections)
            result_filename = f"{image_file.split('.')[0]}_class_result.jpg"
            result_image.save(result_filename)
            print(f"Result image saved as {result_filename}")
    
    except Exception as e:
        print(f"Error in G_Dino class detection: {e}")
    
    # Test pipeline detector
    try:
        print("\nUsing pipeline detector:")
        pipeline_detections = pipeline_detector(image, objects, box_threshold=threshold)
        
        print(f"Detected {len(pipeline_detections)} objects:")
        for det in pipeline_detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")
        
        if pipeline_detections:
            result_image = g_dino.draw_boxes(image.copy(), pipeline_detections)
            result_filename = f"{image_file.split('.')[0]}_pipeline_result.jpg"
            result_image.save(result_filename)
            print(f"Result image saved as {result_filename}")
    
    except Exception as e:
        print(f"Error in pipeline detection: {e}")

if __name__ == "__main__":
    test_pipeline_vs_class()
