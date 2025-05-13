import os
import csv
from PIL import Image
import numpy as np
import yaml

def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))])
    return [
        {
            'image_name': img,
            'image_path': os.path.join(image_dir, img),
            'mask_path': os.path.join(mask_dir, msk)
        }
        for img, msk in zip(image_files, mask_files)
    ]

def compute_metrics(pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_metrics = []
    # accumulators for overall metrics
    total_tp = total_fp = total_fn = total_tn = 0

    for pair in pairs:
        # load ground truth and prediction masks
        gt_name = os.path.basename(pair['image_name'])
        pred_name = os.path.basename(pair['mask_path'])
        gt_path = os.path.join(gt_dir, gt_name)
        gt = np.array(Image.open(gt_path).convert('L')) > 0
        # pred = np.array(Image.open(os.path.join(pred_dir, pair['image_name'])).convert('L')) > 0
        pred_path = os.path.join(pred_dir, pred_name)
        pred = np.array(Image.open(pred_path).convert('L')) > 0

        # compute confusion matrix entries
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()
        tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0

        image_metrics.append({
            'image': pair['image_name'],
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # write image-wise metrics
    image_csv = os.path.join(output_dir, 'metrics_imagewise.csv')
    with open(image_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=image_metrics[0].keys())
        writer.writeheader()
        for row in image_metrics:
            writer.writerow(row)

    # compute overall metrics
    total = total_tp + total_fp + total_fn + total_tn
    overall_accuracy = (total_tp + total_tn) / total if total else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0
    overall_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0

    overall_metrics = {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': total_tn,
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'iou': overall_iou,
    }

    # write overall metrics
    overall_csv = os.path.join(output_dir, 'metrics_overall.csv')
    with open(overall_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in overall_metrics.items():
            writer.writerow([k, v])

    print(f"Image-wise metrics saved to {image_csv}")
    print(f"Overall metrics saved to {overall_csv}")

if __name__ == "__main__":
    # load config
    with open("../config.yml", "r") as f:
        config = yaml.safe_load(f)
    dataset_root = os.path.expanduser(config["segmentation_dataset"])

    # set directories
    # gt_dir = os.path.join(dataset_root, 'train_sim')
    # pred_dir = os.path.join(dataset_root, 'inference_train_sim')
    gt_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/pybullet/output/masks"
    pred_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/inference_train_sim"
    # gt_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/test_real_display"
    # pred_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/real_inference_display"
    gt_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/pybullet/output_like_original_with_mask/masks"
    # pred_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/inference_sim"
    # gt_dir = ""
    pred_dir = "/Users/arthur/Downloads/inference_test_sim_styled"
    out_dir = os.path.join(dataset_root, 'segmentation_metrics_test_sim_styled')

    pairs = get_image_mask_pairs(gt_dir, pred_dir)
    compute_metrics(pairs, out_dir)
