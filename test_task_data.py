from PIL import Image
import random
import os
from florence import G_Dino
from task_od.data import get_task_by_id, get_all_task_ids

def test_with_real_image():
    print("Testing G_Dino with real images from /data/count/train...")
    
    # Get a list of image files from the training directory
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    # Select a random image for testing
    image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, image_file)
    print(f"Using image: {image_path}")
    
    # Load the image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Create G_Dino instance
    g_dino = G_Dino()
    
    # Test with common objects
    objects = ["person", "car", "dog", "cat", "chair", "table"]
    print(f"Testing with objects: {objects}")
    
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
                result_filename = f"real_image_result_{threshold}.jpg"
                result_image.save(result_filename)
                print(f"Result image saved as {result_filename}")
        
        except Exception as e:
            print(f"Error in G_Dino detection: {e}")

def test_with_task_data():
    print("\nTesting G_Dino with task data...")
    
    # Get all task IDs
    task_ids = get_all_task_ids()
    
    # Select a random task
    task_id = random.choice(task_ids)
    task = get_task_by_id(task_id)
    
    print(f"Task ID: {task_id}")
    print(f"Question (objects to detect): {task['question']}")
    print(f"Expected answer (ground truth boxes): {task['answer']}")
    
    # Get a real image instead of the dummy one
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    # Select a random image for testing
    image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, image_file)
    print(f"Using image: {image_path}")
    
    # Load the image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Create G_Dino instance
    g_dino = G_Dino()
    
    # Test with different thresholds
    for threshold in [0.1, 0.2, 0.3]:
        print(f"Using threshold: {threshold}")
        
        try:
            detections = g_dino.detect(image, task['question'], box_threshold=threshold)
            
            print(f"Detected {len(detections)} objects:")
            for det in detections:
                print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")
            
            if detections:
                result_image = g_dino.draw_boxes(image.copy(), detections)
                result_filename = f"task_data_result_{threshold}.jpg"
                result_image.save(result_filename)
                print(f"Result image saved as {result_filename}")
        
        except Exception as e:
            print(f"Error in G_Dino detection: {e}")

if __name__ == "__main__":
    test_with_real_image()
    test_with_task_data()
