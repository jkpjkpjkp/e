from PIL import Image
import random
from florence import G_Dino, pipeline_detector
from task_od.data import get_task_by_id, get_all_task_ids

def create_test_image(width=640, height=480):
    image = Image.new('RGB', (width, height), (255, 255, 255))
    return image

def test_g_dino_class():
    print("Testing G_Dino class implementation...")

    task_ids = get_all_task_ids()
    task_id = random.choice(task_ids)
    task = get_task_by_id(task_id)

    image = create_test_image(task['width'], task['height'])
    objects = task['question']

    g_dino = G_Dino()

    try:
        detections = g_dino.detect(image, objects)

        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")

        if detections:
            result_image = g_dino.draw_boxes(image.copy(), detections)
            result_image.save("g_dino_result.jpg")
            print("Result image saved as g_dino_result.jpg")

        return len(detections) > 0
    except Exception as e:
        print(f"Error in G_Dino class detection: {e}")
        return False

def test_pipeline_detector():
    print("\nTesting pipeline detector implementation...")

    task_ids = get_all_task_ids()
    task_id = random.choice(task_ids)
    task = get_task_by_id(task_id)

    image = create_test_image(task['width'], task['height'])
    objects = task['question']

    try:
        detections = pipeline_detector(image, objects)

        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(f"- {det['label']}: score {det['score']:.2f}, box {det['box']}")

        return len(detections) > 0
    except Exception as e:
        print(f"Error in pipeline detection: {e}")
        return False

if __name__ == "__main__":
    print("Testing G_Dino implementations...")

    class_success = test_g_dino_class()
    pipeline_success = test_pipeline_detector()

    print("\nTest Summary:")
    print(f"G_Dino class implementation: {'SUCCESS' if class_success else 'FAILED'}")
    print(f"Pipeline implementation: {'SUCCESS' if pipeline_success else 'FAILED'}")
