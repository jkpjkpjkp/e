from PIL import Image
import os
import random
from florence import G_Dino

def test_g_dino_with_real_images():
    print("Testing G_Dino with real images from /data/count/train...")
    image_dir = "/data/count/train"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if not image_files:
        print("No image files found in the directory")
        return

    selected_images = random.sample(image_files, min(5, len(image_files)))
    g_dino = G_Dino()
    objects = ["person", "car", "dog", "cat", "chair", "table", "bicycle", "bird", "boat"]
    thresholds = [0.1, 0.2, 0.3, 0.4]

    for image_file in selected_images:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nTesting with image: {image_path}")
        image = Image.open(image_path)
        print(f"Image size: {image.size}")

        for threshold in thresholds:
            print(f"\nUsing threshold: {threshold}")
            try:
                detections = g_dino.detect(image, objects, box_threshold=threshold)
                print(f"Detected {len(detections)} objects:")
                for det in detections:
                    print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")

                if detections:
                    result_image = g_dino.draw_boxes(image.copy(), detections)
                    result_filename = f"{image_file.split('.')[0]}_threshold_{threshold}.jpg"
                    result_image.save(result_filename)
                    print(f"Result image saved as {result_filename}")
            except Exception as e:
                print(f"Error in G_Dino detection: {e}")

if __name__ == "__main__":
    test_g_dino_with_real_images()
