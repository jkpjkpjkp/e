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

def calculate_ap(pred, gt_boxes, iou_threshold):
    if not gt_boxes:
        return 1.0 if not pred else 0.0
    pred = sorted(pred, key=lambda x: x['score'], reverse=True)
    detected = [False] * len(gt_boxes)
    tp = [0] * len(pred)
    fp = [0] * len(pred)
    for idx, pred in enumerate(pred):
        best_iou = 0
        best_gt_idx = None
        for i, gt in enumerate(gt_boxes):
            if detected[i]:
                continue
            iou = IoU(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx is not None:
            tp[idx] = 1
            detected[best_gt_idx] = True
        else:
            fp[idx] = 1
    tp_count = np.cumsum(tp)
    fp_count = np.cumsum(fp)
    recalls = tp_count / len(gt_boxes)
    precisions = tp_count / (tp_count + fp_count + 1e-6)
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

def compute_coco_mAP(predictions, gt_boxes, gt_label, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    pred = [p for p in predictions if p['label'] == gt_label]
    ap_thresholds = []
    for iou_threshold in iou_thresholds:
        ap = calculate_ap(pred, gt_boxes, iou_threshold)
        ap_thresholds.append(ap)
    return np.mean(ap_thresholds)

def get_task_by_id(id):
    id = int(id)
    ret = df[id].to_dict()
    ret['image'] = Image.open(ret['image_path'][0])
    ret['question'] = ret['label'][0]
    ret['answer'] = list(ret['annotations'][0])  # List of dicts with 'bbox' and 'class'
    ret['answer'] = [list(x) for x in ret['answer']]
    assert isinstance(ret['answer'], list)
    assert isinstance(ret['answer'][0], list)
    assert len(ret['answer'][0]) == 4
    ret['id'] = id
    ret['score'] = lambda predictions: compute_coco_mAP(predictions, ret['answer'], ret['question'])
    return ret