from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict

def parse_yolo_label(label_path):
    """Parse YOLO format label file and return list of detections."""
    detections = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    detections.append({
                        'class': cls_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    return detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes in normalized coordinates."""
    # Convert from center coordinates to corner coordinates
    def center_to_corners(box):
        x1 = box['x_center'] - box['width'] / 2
        y1 = box['y_center'] - box['height'] / 2
        x2 = box['x_center'] + box['width'] / 2
        y2 = box['y_center'] + box['height'] / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = center_to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = center_to_corners(box2)
    
    # Calculate intersection
    x_inter_left = max(x1_1, x1_2)
    y_inter_top = max(y1_1, y1_2)
    x_inter_right = min(x2_1, x2_2)
    y_inter_bottom = min(y2_1, y2_2)
    
    if x_inter_right < x_inter_left or y_inter_bottom < y_inter_top:
        return 0.0
    
    inter_area = (x_inter_right - x_inter_left) * (y_inter_bottom - y_inter_top)
    
    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth detections and return metrics."""
    matched_predictions = set()
    matched_gt = set()
    true_positives = 0
    
    # Match predictions to ground truth
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            # Check if class matches
            if pred['class'] == ground_truth[best_gt_idx]['class']:
                true_positives += 1
                matched_predictions.add(pred_idx)
                matched_gt.add(best_gt_idx)
    
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)
    
    return true_positives, false_positives, false_negatives

def test_model(model_path="runs/detect/train/weights/best.pt", test_dir="test", conf_threshold=0.25):
    """Test model on all images in test directory and compare with ground truth labels."""
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get image and label directories
    images_dir = os.path.join(test_dir, "images")
    labels_dir = os.path.join(test_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(images_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nTesting on {len(image_files)} images...")
    print("=" * 80)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    image_results = []
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)
        label_base = os.path.splitext(image_file)[0]
        label_path = os.path.join(labels_dir, f"{label_base}.txt")
        
        # Get ground truth
        ground_truth = parse_yolo_label(label_path)
        
        # Run inference
        results = model(image_path, conf=conf_threshold, verbose=False)
        r = results[0]
        
        # Extract predictions
        predictions = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x_center = float(box.xywhn[0][0])  # normalized x
            y_center = float(box.xywhn[0][1])  # normalized y
            width = float(box.xywhn[0][2])     # normalized width
            height = float(box.xywhn[0][3])    # normalized height
            conf = float(box.conf[0])
            
            predictions.append({
                'class': cls_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'confidence': conf
            })
        
        # Match predictions to ground truth
        tp, fp, fn = match_predictions_to_ground_truth(predictions, ground_truth)

        failure_score = fp + fn

        image_results.append({
            "image": image_file,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "failure_score": failure_score,
            "ground_truth": len(ground_truth),
            "predictions": len(predictions)
        })
        
        # Update global stats
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Update per-class stats
        matched_gt = set()

        # Check predictions
        for pred in predictions:
            pred_class = pred['class']

            matched = False

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                if pred_class != gt['class']:
                    continue

                iou = calculate_iou(pred, gt)

                if iou >= 0.5:
                    matched = True
                    matched_gt.add(gt_idx)
                    break

            if matched:
                class_stats[pred_class]['tp'] += 1
            else:
                class_stats[pred_class]['fp'] += 1

        # Any unmatched ground truth = false negative
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in matched_gt:
                class_stats[gt['class']]['fn'] += 1
        
        # Print progress
        if idx % 10 == 0 or idx == len(image_files):
            print(f"[{idx}/{len(image_files)}] {image_file}")
            print(f"  Ground Truth: {len(ground_truth)} | Predictions: {len(predictions)} | TP: {tp}, FP: {fp}, FN: {fn}")
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total True Positives:  {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Load class names from data.yaml
    if os.path.exists("data.yaml"):
        with open("data.yaml", 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', {})
            if isinstance(class_names, list):
                class_names = {i: name for i, name in enumerate(class_names)}
    else:
        class_names = {}
    
    print("\n" + "=" * 80)
    print("PER-CLASS STATISTICS")
    print("=" * 80)
    
    for class_id in sorted(class_stats.keys()):
        stats = class_stats[class_id]
        class_name = class_names.get(class_id, f"Class {class_id}")
        class_tp = stats['tp']
        class_fp = stats['fp']
        class_fn = stats['fn']
        
        class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
        class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
        
        print(f"{class_name:10} | TP: {class_tp:3} FP: {class_fp:3} FN: {class_fn:3} | Precision: {class_precision:.4f} | Recall: {class_recall:.4f}")

    # Find worst failure cases
    worst_cases = sorted(
        image_results,
        key=lambda x: x["failure_score"],
        reverse=True
    )

    print("\n" + "=" * 80)
    print("TOP 3 FAILURE CASES")
    print("=" * 80)

    for i, case in enumerate(worst_cases[:3], 1):
        print(f"\n#{i}")
        print(f"Image: {case['image']}")
        print(f"Failure Score: {case['failure_score']}")
        print(f"TP: {case['tp']} | FP: {case['fp']} | FN: {case['fn']}")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Predictions: {case['predictions']}")
if __name__ == "__main__":
    test_model()
