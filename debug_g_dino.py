from PIL import Image
from florence import G_Dino
import os

def debug_g_dino():
    print("Debugging G_Dino with real images...")
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if not image_files:
        print("No image files found in the directory")
        return

    image_path = os.path.join(image_dir, image_files[0])
    print(f"Using image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    g_dino = G_Dino()
    objects_to_test = [
        ["person"],
        ["car"],
        ["dog"],
        ["person", "car", "dog"],
    ]
    thresholds = [0.1, 0.2, 0.3, 0.4]

    for objects in objects_to_test:
        print(f"\nTesting with objects: {objects}")
        for threshold in thresholds:
            print(f"  Using threshold: {threshold}")
            try:
                detections = g_dino.detect(image, objects, box_threshold=threshold)
                print(f"  Detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['label']}: score {det['score']:.2f}, box {det['box']}")

                if detections:
                    result_image = g_dino.draw_boxes(image.copy(), detections)
                    result_filename = f"g_dino_result_{'-'.join(objects)}_{threshold}.jpg"
                    result_image.save(result_filename)
                    print(f"  Result image saved as {result_filename}")
            except Exception as e:
                print(f"  Error in G_Dino detection: {e}")

if __name__ == "__main__":
    debug_g_dino()
