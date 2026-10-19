"""Evaluate a trained model on one split and store the numbers as JSON.

Reports the Ultralytics mAP values plus keypoint MAE, angle error and FPS.

    python -m erc_pose.evaluate --weights runs/pose/weights/best.pt --data data/processed/pose/data.yaml --name pose
"""

import argparse
import json
from pathlib import Path

import numpy as np

from erc_pose.benchmark import available_devices, list_images, measure_fps, train_imgsz
from erc_pose.direction import apply_direction, load_direction_model
from erc_pose.metrics import angle_stats, keypoint_mae, match_boxes
from erc_pose.orientation import angle_from_keypoints, angular_error
from erc_pose.predict import probes_from_result

ANGLE_TOLERANCE_DEG = 5.0  # target accuracy from the proposal


def read_ground_truth(label_path, width, height):
    """Boxes (N, 4) xyxy and keypoints (N, 2, 3) in pixels; keypoints are NaN when not labelled."""
    boxes, kpts = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = [float(v) for v in line.split()]
            if len(values) < 5:
                continue
            cx, cy, w, h = values[1:5]
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
            if len(values) >= 11:
                k = np.asarray(values[5:11]).reshape(2, 3) * [width, height, 1]
            else:
                k = np.full((2, 3), np.nan)
            kpts.append(k)
    return np.asarray(boxes).reshape(-1, 4), np.asarray(kpts).reshape(-1, 2, 3)


def orientation_metrics(model, gt_dir, split, device, imgsz=640, conf=0.25, direction_model=None, tta=False):
    """Match predictions to ground truth and compute recall, keypoint MAE and angle error.

    With `direction_model` (a pose model or a direction classifier), OBB angles get their direction from it.
    """
    n_gt = n_pred = n_matched = 0
    pred_kpts, gt_kpts, pred_angles, gt_angles, dir_pred, dir_gt = [], [], [], [], [], []
    images = list_images(Path(gt_dir) / "images" / split)
    for image in images:
        result = model.predict(str(image), device=device, imgsz=imgsz, conf=conf, verbose=False)[0]
        height, width = result.orig_shape
        gt_boxes, gt_k = read_ground_truth(Path(gt_dir) / "labels" / split / f"{image.stem}.txt", width, height)
        probes = probes_from_result(result)
        if direction_model is not None:
            probes = apply_direction(direction_model, result.orig_img, probes, conf, device, tta)
        n_gt, n_pred = n_gt + len(gt_boxes), n_pred + len(probes)
        if not probes or not len(gt_boxes):
            continue
        matches = match_boxes([p.xyxy for p in probes], [p.conf for p in probes], gt_boxes)
        n_matched += len(matches)
        for p, g in matches:
            probe = probes[p]
            if not (gt_k[g, :, 2] > 0).all():  # also false for NaN, i.e. no keypoint labels
                continue
            if probe.kpts is not None and probe.directed:  # undirected axis ends have no top/bottom order
                pred_kpts.append(probe.kpts)
                gt_kpts.append(gt_k[g, :, :2])
            if probe.angle_deg is not None:
                gt_angle = float(angle_from_keypoints(gt_k[g, 0, :2], gt_k[g, 1, :2]))
                pred_angles.append(probe.angle_deg)
                gt_angles.append(gt_angle)
                if probe.directed:
                    dir_pred.append(probe.angle_deg)
                    dir_gt.append(gt_angle)

    out = {
        "n_images": len(images),
        "n_gt": n_gt,
        "recall": n_matched / n_gt if n_gt else float("nan"),
        "precision": n_matched / n_pred if n_pred else float("nan"),
        "kpt_mae_px": keypoint_mae(pred_kpts, gt_kpts),
    }
    # the axis error (mod 180) is comparable between pose and obb models
    axis = angle_stats(pred_angles, gt_angles, 180.0, ANGLE_TOLERANCE_DEG)
    out.update(angle_axis_mae_deg=axis["mae_deg"], angle_axis_median_deg=axis["median_deg"],
               angle_axis_within_5deg=axis["within_tol"])
    if dir_pred:  # full 360-degree error, over the probes whose direction is known
        full = angle_stats(dir_pred, dir_gt, 360.0, ANGLE_TOLERANCE_DEG)
        out.update(angle_mae_deg=full["mae_deg"], angle_median_deg=full["median_deg"],
                   angle_within_5deg=full["within_tol"], directed_share=len(dir_pred) / len(pred_angles),
                   direction_acc=float((angular_error(dir_pred, dir_gt) < 90).mean()))
    return out


def evaluate(weights, data, name, split="test", gt_dir=None, device=None, imgsz=None, conf=0.25,
             out_dir="results", fps=True, direction_weights=None, direction_tta=False):
    from ultralytics import YOLO

    data = Path(data).resolve()
    if gt_dir is None:  # the pose dataset holds boxes and keypoints for the same images
        pose_dir = data.parent.parent / "pose"
        gt_dir = pose_dir if pose_dir.exists() else data.parent
    model = YOLO(weights)
    imgsz = imgsz or train_imgsz(model)
    if device is None:
        device = available_devices()[-1]

    out = {"name": name, "task": model.task, "weights": str(weights), "split": split, "imgsz": imgsz,
           "params_m": sum(p.numel() for p in model.model.parameters()) / 1e6}
    val = model.val(data=str(data), split=split, device=device, imgsz=imgsz, plots=False, verbose=False,
                    project=str(Path(out_dir).resolve() / "val"), name=name, exist_ok=True)
    out.update(map50=float(val.box.map50), map50_95=float(val.box.map))
    if model.task == "pose":
        out.update(pose_map50=float(val.pose.map50), pose_map50_95=float(val.pose.map))
    direction_model = load_direction_model(direction_weights) if direction_weights else None
    if direction_model is not None:
        out.update(direction_weights=str(direction_weights), direction_tta=direction_tta)
    out.update(orientation_metrics(model, gt_dir, split, device, imgsz, conf, direction_model, direction_tta))
    if fps:
        images = list_images(Path(gt_dir) / "images" / split)
        for dev in available_devices():
            out["fps_cpu" if dev == "cpu" else "fps_gpu"] = measure_fps(model, images, dev, imgsz,
                                                                         direction_model=direction_model,
                                                                         tta=direction_tta)

    path = Path(out_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="data.yaml the model was trained on")
    parser.add_argument("--name", required=True, help="row name in the comparison table")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--gt", help="dataset folder with ground-truth labels (default: the pose dataset)")
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int, help="default: the training image size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--direction-weights",
                        help="pose model or direction classifier that gives OBB angles their direction")
    parser.add_argument("--direction-tta", action="store_true", help="flip test-time augmentation for the direction")
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.name, args.split, args.gt, args.device, args.imgsz, args.conf,
             args.out, not args.no_fps, args.direction_weights, args.direction_tta)


if __name__ == "__main__":
    main()
