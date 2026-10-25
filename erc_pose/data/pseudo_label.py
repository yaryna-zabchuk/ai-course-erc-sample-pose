"""Pseudo-label unlabelled frames, keeping only confident ones.

    python -m erc_pose.data.pseudo_label --weights runs/pose/weights/best.pt \
        --obb-weights runs/obb/weights/best.pt --images data/raw/erc-sample-pose/data --out data/pseudo
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from erc_pose.benchmark import IMAGE_SUFFIXES, train_imgsz
from erc_pose.data.split import DEFAULT_SESSION_GAP_S, frame_time
from erc_pose.predict import fuse_direction, probes_from_result


def held_out_times(dataset_dir):
    """{source: [timestamps]} of the val and test frames, and the set of all labelled stems."""
    times, labelled = defaultdict(list), set()
    for path in Path(dataset_dir).glob("images/*/*"):
        labelled.add(path.stem)
        parsed = frame_time(path.stem)
        if parsed and path.parent.name in ("val", "test"):
            times[parsed[0]].append(parsed[1])
    return times, labelled


def is_near_held_out(stem, times, gap_s):
    parsed = frame_time(stem)
    if parsed is None:
        return False
    source, time = parsed
    return any(abs(time - t) < gap_s for t in times.get(source, ()))


def pseudo_label(weights, image_dirs, dataset_dir, out_dir, high=0.6, low=0.1,
                 gap_s=DEFAULT_SESSION_GAP_S, device=None, obb_weights=None, max_disagreement=20.0):
    """Write confident predictions as YOLO Pose labels."""
    from ultralytics import YOLO

    pose = YOLO(weights)
    if pose.task != "pose":
        raise ValueError("pseudo-labelling needs a pose model")
    obb = YOLO(obb_weights) if obb_weights else None
    times, labelled = held_out_times(dataset_dir)
    images = sorted({p.stem: p for d in image_dirs for p in Path(d).rglob("*")
                     if p.suffix.lower() in IMAGE_SUFFIXES}.values())

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir()
    counts = defaultdict(int)
    for path in images:
        if path.stem in labelled:
            counts["already labelled"] += 1
            continue
        if is_near_held_out(path.stem, times, gap_s):
            counts["near val/test"] += 1
            continue
        result = pose.predict(str(path), imgsz=train_imgsz(pose), conf=low, device=device, verbose=False)[0]
        probes = probes_from_result(result)
        if obb is not None:
            axes = obb.predict(str(path), imgsz=train_imgsz(obb), conf=low, device=device, verbose=False)[0]
            probes = fuse_direction(probes_from_result(axes), probes)
            agree = [p.directed and p.agreement_deg <= max_disagreement for p in probes]
        else:
            agree = [True] * len(probes)
        if any(p.conf < high or not ok for p, ok in zip(probes, agree)):
            counts["uncertain"] += 1
            continue
        height, width = result.orig_shape
        lines = []
        for probe in probes:
            # line-extent box, build_dataset widens it later
            (x1, y1), (x2, y2) = probe.kpts.min(0), probe.kpts.max(0)
            box = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
            points = np.clip(probe.kpts / [width, height], 0, 1).ravel()
            lines.append("0 " + " ".join(f"{v:.6f}" for v in box) + f" {points[0]:.6f} {points[1]:.6f} 2"
                         f" {points[2]:.6f} {points[3]:.6f} 2")
        counts["with probes" if lines else "background"] += 1
        shutil.copy2(path, out_dir / "images" / path.name)
        (out_dir / "labels" / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(", ".join(f"{k}: {v}" for k, v in counts.items()))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="trained pose model")
    parser.add_argument("--images", nargs="+", required=True, help="folders with unlabelled frames")
    parser.add_argument("--dataset", default="data/processed/pose", help="built dataset with the human labels")
    parser.add_argument("--out", default="data/pseudo")
    parser.add_argument("--high", type=float, default=0.6, help="keep detections at or above this confidence")
    parser.add_argument("--low", type=float, default=0.1, help="skip frames with detections between low and high")
    parser.add_argument("--gap", type=float, default=DEFAULT_SESSION_GAP_S, help="seconds kept away from val/test frames")
    parser.add_argument("--obb-weights", help="OBB model for the axis; the pose model then only gives the direction")
    parser.add_argument("--max-disagreement", type=float, default=20.0,
                        help="with --obb-weights: max angle between pose direction and OBB axis")
    parser.add_argument("--device")
    args = parser.parse_args()
    pseudo_label(args.weights, args.images, args.dataset, args.out, args.high, args.low,
                 args.gap, args.device, args.obb_weights, args.max_disagreement)


if __name__ == "__main__":
    main()
