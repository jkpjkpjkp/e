from PIL import Image
import os
import random
from florence import G_Dino

def test_with_real_images():
    print("Testing G_Dino with real images from /data/count/train...")
    
    # Get a list of image files from the training directory
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    # Select 3 random images for testing
    selected_images = random.sample(image_files, min(3, len(image_files)))
    
    # Create G_Dino instance
    g_dino = G_Dino()
    
    # Common objects to detect
    objects = ["person", "car", "dog", "cat", "chair", "table", "bicycle", "bird", "boat"]
    
    for image_file in selected_images:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nTesting with image: {image_path}")
        
        # Load the image
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        
        # Try different thresholds
        for threshold in [0.1, 0.2, 0.3]:
            print(f"Using threshold: {threshold}")
            
            try:
                detections = g_dino.detect(image, objects, box_threshold=threshold)
                
                print(f"Detected {len(detections)} objects:")
                for det in detections:
                    print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")
                
                if detections:
                    result_image = g_dino.draw_boxes(image.copy(), detections)
                    result_filename = f"{image_file.split('.')[0]}_result_{threshold}.jpg"
                    result_image.save(result_filename)
                    print(f"Result image saved as {result_filename}")
            
            except Exception as e:
                print(f"Error in G_Dino detection: {e}")

if __name__ == "__main__":
    test_with_real_images()
