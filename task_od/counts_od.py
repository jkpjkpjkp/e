import polars as pl
from PIL import Image
import random
import numpy as np

df = pl.read_parquet('dataset_grouped.parquet')

def get_all_task_ids():
    return list(range(len(df)))

def IoU(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    intersection = (x2 - x1) * (y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - intersection
    return intersection / union

def calculate_ap(pred_for_class, gt_boxes, iou_threshold):
    if not gt_boxes:
        return 1.0 if not pred_for_class else 0.0
    pred_for_class = sorted(pred_for_class, key=lambda x: x['confidence'], reverse=True)
    detected = [False] * len(gt_boxes)
    tp = [0] * len(pred_for_class)
    fp = [0] * len(pred_for_class)
    for idx, pred in enumerate(pred_for_class):
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if detected[i]:
                continue
            iou = IoU(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[idx] = 1
            detected[best_gt_idx] = True
        else:
            fp[idx] = 1
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    ap = 0
    for t in np.arange(0, 1.01, 0.01):
        if np.any(recalls >= t):
            p = np.max(precisions[recalls >= t])
        else:
            p = 0
        ap += p / 101
    return ap

def compute_coco_mAP(predictions, ground_truths, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    gt_by_class = {}
    for gt in ground_truths:
        cls = gt['class']
        if cls not in gt_by_class:
            gt_by_class[cls] = []
        gt_by_class[cls].append(gt)
    ap_by_class = {}
    for cls in gt_by_class:
        gt_boxes = gt_by_class[cls]
        pred_for_class = [p for p in predictions if p['class'] == cls]
        ap_thresholds = []
        for iou_threshold in iou_thresholds:
            ap = calculate_ap(pred_for_class, gt_boxes, iou_threshold)
            ap_thresholds.append(ap)
        ap_by_class[cls] = np.mean(ap_thresholds)
    return ap_by_class  # Return dictionary of APs for each class

def get_task_by_id(id):
    id = int(id)
    ret = df[id].to_dict()
    ret['image'] = Image.open(ret['image_path'][0])
    ret['question'] = ret['label'][0]
    ret['answer'] = ret['annotations'][0]  # List of dicts with 'bbox' and 'class'
    ret['id'] = id
    ret['score'] = lambda predictions: compute_coco_mAP(predictions, ret['answer'])
    return ret

def get_dummy_task():
    image = Image.new('RGB', (540, 540), (255, 255, 255))
    num_objects = random.randint(1, 5)
    ground_truths = []
    for i in range(num_objects):
        x1, y1 = random.randint(0, 540), random.randint(0, 540)
        x2, y2 = random.randint(0, 540), random.randint(0, 540)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        cls = random.randint(0, 2)  # 3 classes
        ground_truths.append({'bbox': (x1, y1, x2, y2), 'class': cls})
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image.paste((Image.new('RGB', (x2 - x1, y2 - y1), color), (x1, y1)))
    return {
        'image': image,
        'question': 'What are the bounding boxes and classes of the objects in the image?',
        'answer': ground_truths,
        'id': f'dummy_{"_".join(str(gt["bbox"]) for gt in ground_truths)}',
        'score': lambda predictions: compute_coco_mAP(predictions, ground_truths),
    }