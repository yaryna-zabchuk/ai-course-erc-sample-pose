"""The same test frames through several models, one column per model.

    python scripts/compare_grid.py --out results/v2/compare.jpg --models pose obb seg unet twostage rcnn classic

Each cell is a crop around the labelled probes: the label in green, the prediction in red,
with the axis error of the matched probe.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from erc_pose.benchmark import list_images  # noqa: E402
from erc_pose.evaluate import read_ground_truth  # noqa: E402
from erc_pose.metrics import match_boxes  # noqa: E402
from erc_pose.models import load_predictor  # noqa: E402
from erc_pose.orientation import angle_from_keypoints, angular_error  # noqa: E402

CELL = (360, 240)


def pick_frames(gt, count, seed):
    """Frames with one or two probes, half of them close-ups when there are any."""
    images = [p for p in list_images(Path(gt) / "images" / "test")
              if 1 <= len((Path(gt) / "labels" / "test" / f"{p.stem}.txt").read_text().split("\n")) <= 2
              and (Path(gt) / "labels" / "test" / f"{p.stem}.txt").read_text().strip()]
    rng = np.random.default_rng(seed)
    close = [p for p in images if "19700101" in p.stem]
    far = [p for p in images if p not in close]
    picked = list(rng.choice(close, min(len(close), count // 2), replace=False)) if close else []
    picked += list(rng.choice(far, count - len(picked), replace=False))
    return picked


def crop_box(boxes, shape):
    x1, y1 = boxes[:, :2].min(0)
    x2, y2 = boxes[:, 2:].max(0)
    side = max((x2 - x1) * 1.8, (y2 - y1) * 1.8 * CELL[0] / CELL[1], 160)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = side, side * CELL[1] / CELL[0]
    left = int(np.clip(cx - w / 2, 0, max(shape[1] - w, 0)))
    top = int(np.clip(cy - h / 2, 0, max(shape[0] - h, 0)))
    return left, top, int(min(w, shape[1])), int(min(h, shape[0]))


def cell(frame, gt_boxes, gt_kpts, probes, region):
    image = frame.copy()
    for k in gt_kpts:
        cv2.line(image, tuple(map(int, k[0, :2])), tuple(map(int, k[1, :2])), (0, 220, 0), 2, cv2.LINE_AA)
    errors = []
    if probes:
        for p, g in match_boxes([p.xyxy for p in probes], [p.conf for p in probes], gt_boxes):
            if probes[p].angle_deg is not None:
                gt_angle = angle_from_keypoints(gt_kpts[g, 0, :2], gt_kpts[g, 1, :2])
                errors.append(float(angular_error(probes[p].angle_deg, gt_angle, 180.0)))
    for probe in probes:
        if probe.kpts is not None:
            cv2.line(image, tuple(map(int, probe.kpts[0])), tuple(map(int, probe.kpts[1])), (0, 0, 255), 2,
                     cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = map(int, probe.xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    left, top, w, h = region
    out = cv2.resize(image[top:top + h, left:left + w], CELL, interpolation=cv2.INTER_AREA)
    missed = len(gt_boxes) - len(errors) if probes else len(gt_boxes)
    text = " ".join(f"{e:.1f}deg" for e in errors) + (f" missed {missed}" if missed > 0 else "")
    cv2.rectangle(out, (0, CELL[1] - 26), (CELL[0], CELL[1]), (0, 0, 0), -1)
    cv2.putText(out, text.strip() or "-", (6, CELL[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", default="data/processed_v2/pose")
    parser.add_argument("--models", nargs="+", default=["pose", "obb", "seg", "unet", "twostage", "rcnn", "classic"])
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="results/v2/compare.jpg")
    args = parser.parse_args()

    frames = pick_frames(args.gt, args.frames, args.seed)
    data = []
    for path in frames:
        image = cv2.imread(str(path))
        boxes, kpts = read_ground_truth(Path(args.gt) / "labels" / "test" / f"{path.stem}.txt", image.shape[1],
                                        image.shape[0])
        data.append((image, boxes, kpts, crop_box(boxes, image.shape)))
    columns = []
    for name in args.models:
        predict = load_predictor(name)
        column = [cell(image, boxes, kpts, predict(image), region) for image, boxes, kpts, region in data]
        title = np.full((34, CELL[0], 3), 255, np.uint8)
        cv2.putText(title, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        columns.append(np.vstack([title] + column))
        print(name, flush=True)
    grid = np.hstack([np.hstack([c, np.full((c.shape[0], 4, 3), 255, np.uint8)]) for c in columns])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(args.out)


if __name__ == "__main__":
    main()
