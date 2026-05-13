import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hausdorff import hausdorff_distance
from medpy import metric
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from Model.model import KnowSAM
from utils.training_monitor import save_evaluation_artifacts


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_multiclass")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=10)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", type=str, default="sam_adpt")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--thd", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multimask", type=bool, default=False)
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--SGDL_model_path", type=str,
                        default="./Results/Multiclass_KnowSAM_V100_106_117_13_13/SGDL_best_model.pth")
    parser.add_argument("--save_dir", type=str,
                        default="./Results/Multiclass_KnowSAM_V100_106_117_13_13/prediction_test")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def setup_logger(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "prediction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def to_uint8_rgb(image_tensor: torch.Tensor):
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, bgr in CLASS_COLORS.items():
        color[mask == class_idx] = bgr
    return color


def overlay_multiclass(image_bgr: np.ndarray, mask: np.ndarray, alpha=0.35):
    color = colorize_mask(mask)
    fg = mask > 0
    overlay = image_bgr.copy()
    blended = cv2.addWeighted(image_bgr, 1.0, color, alpha, 0)
    overlay[fg] = blended[fg]
    return overlay


def class_metrics(gt: np.ndarray, pred: np.ndarray, class_idx: int):
    gt_c = gt == class_idx
    pred_c = pred == class_idx
    gt_sum = gt_c.sum()
    pred_sum = pred_c.sum()
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 0.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, float("nan")
    return (
        float(metric.binary.dc(pred_c, gt_c)),
        float(metric.binary.jc(pred_c, gt_c)),
        float(hausdorff_distance(gt_c.astype(np.uint8), pred_c.astype(np.uint8)) * 0.95),
    )


def evaluate_multiclass(gt: np.ndarray, pred: np.ndarray, num_classes: int):
    result = {}
    rows = []
    for class_idx in range(1, num_classes):
        dice, iou, hd95 = class_metrics(gt, pred, class_idx)
        result[f"class_{class_idx}_dice"] = dice
        result[f"class_{class_idx}_iou"] = iou
        result[f"class_{class_idx}_hd95"] = hd95
        rows.append((dice, iou, hd95))
    metrics_array = np.array(rows, dtype=np.float32)
    result["dice"] = float(np.nanmean(metrics_array[:, 0]))
    result["iou"] = float(np.nanmean(metrics_array[:, 1]))
    result["hd95"] = float(np.nanmean(metrics_array[:, 2]))
    return result


def save_case_outputs(save_dir: Path, case_name: str, ori_image: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    dirs = {
        "original": save_dir / "original",
        "gt_mask": save_dir / "gt_mask",
        "pred_mask": save_dir / "pred_mask",
        "gt_color": save_dir / "gt_color",
        "pred_color": save_dir / "pred_color",
        "overlay": save_dir / "overlay",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dirs["original"] / case_name), ori_image)
    cv2.imwrite(str(dirs["gt_mask"] / case_name), gt.astype(np.uint8))
    cv2.imwrite(str(dirs["pred_mask"] / case_name), pred.astype(np.uint8))
    cv2.imwrite(str(dirs["gt_color"] / case_name), colorize_mask(gt))
    cv2.imwrite(str(dirs["pred_color"] / case_name), colorize_mask(pred))
    cv2.imwrite(str(dirs["overlay"] / case_name), overlay_multiclass(ori_image, pred))


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)

    transforms = build_transforms(args)
    test_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split=args.split,
                                 transform=transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    model = KnowSAM(args, bilinear=False).to(args.device)
    checkpoint = torch.load(args.SGDL_model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()

    case_metrics = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            image = batch["image"].to(args.device)
            label = batch["label"].to(args.device)
            case_name = batch["case_name"][0]
            ori_image = to_uint8_rgb(batch["ori_image"][0]).copy()

            _, _, _, _, fusion_map = model(image)
            pred = torch.argmax(torch.softmax(fusion_map, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            gt = label.squeeze(0).cpu().numpy().astype(np.uint8)
            metrics = evaluate_multiclass(gt, pred, args.num_classes)
            save_case_outputs(save_dir, case_name, ori_image, gt, pred)

            row = {
                "index": index,
                "case_name": case_name,
                **metrics,
                "pred_class_1_pixels": int((pred == 1).sum()),
                "pred_class_2_pixels": int((pred == 2).sum()),
                "gt_class_1_pixels": int((gt == 1).sum()),
                "gt_class_2_pixels": int((gt == 2).sum()),
            }
            case_metrics.append(row)
            logging.info("case=%s dice=%.6f iou=%.6f hd95=%.6f", case_name, row["dice"], row["iou"], row["hd95"])

    valid = [row for row in case_metrics if not np.isnan(row["dice"])]
    summary = {
        "split": args.split,
        "model_path": os.path.abspath(args.SGDL_model_path),
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "avg_dice": float(np.nanmean([row["dice"] for row in valid])) if valid else float("nan"),
        "avg_iou": float(np.nanmean([row["iou"] for row in valid])) if valid else float("nan"),
        "avg_hd95": float(np.nanmean([row["hd95"] for row in valid])) if valid else float("nan"),
    }
    for class_idx in range(1, args.num_classes):
        summary[f"class_{class_idx}_avg_dice"] = float(np.nanmean([row[f"class_{class_idx}_dice"] for row in case_metrics]))
        summary[f"class_{class_idx}_avg_iou"] = float(np.nanmean([row[f"class_{class_idx}_iou"] for row in case_metrics]))
        summary[f"class_{class_idx}_avg_hd95"] = float(np.nanmean([row[f"class_{class_idx}_hd95"] for row in case_metrics]))

    csv_path = save_dir / "case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(case_metrics[0].keys()) if case_metrics else [])
        writer.writeheader()
        writer.writerows(case_metrics)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "cases": case_metrics}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    save_evaluation_artifacts(
        save_dir,
        case_metrics,
        summary,
        metadata={
            "split": args.split,
            "model_path": os.path.abspath(args.SGDL_model_path),
            "log_path": str(log_path.resolve()),
            "root_summary_path": str(summary_path.resolve()),
            "root_case_metrics_path": str(csv_path.resolve()),
        },
    )

    logging.info("Prediction summary: %s", summary)
    print(f"{args.split} :")
    print("avg_dice: ", summary["avg_dice"])
    print("avg_iou: ", summary["avg_iou"])
    print("avg_hd95: ", summary["avg_hd95"])
    print("outputs saved to:", save_dir.resolve())


if __name__ == "__main__":
    main()
