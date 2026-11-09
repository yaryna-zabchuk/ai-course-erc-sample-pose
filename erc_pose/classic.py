"""Classical baseline without learning: green colour threshold + shape filter + rotated rectangle.

    python -m erc_pose.classic --gt data/processed_v2/pose --name classic_v2
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from erc_pose.benchmark import list_images
from erc_pose.data.build_dataset import DEFAULT_WIDTH_RATIO
from erc_pose.orientation import angle_from_keypoints
from erc_pose.predict import Probe, axis_ends


@dataclass
class Params:
    h_lo: int = 55        # OpenCV hue, 0..179; the probes are at about 60-80
    h_hi: int = 85
    s_min: int = 110
    v_min: int = 60
    min_len: float = 12   # px, long side of the rotated rectangle
    min_elong: float = 3  # long / short side


def detect(image, p=Params()):
    """Probes in a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (p.h_lo, p.s_min, p.v_min), (p.h_hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    probes = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if len(contour) < 3:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        length, width = max(w, h), max(min(w, h), 1.0)
        if length < p.min_len or length / width < p.min_elong:
            continue
        ends = axis_ends([cx, cy, w, h], angle if w >= h else angle + 90)
        # box drawn the same way as the training boxes: width_ratio x length around the axis
        d = (ends[1] - ends[0]) / length
        half = np.abs(d) * length / 2 + np.abs(d[::-1]) * DEFAULT_WIDTH_RATIO * length / 2
        xyxy = np.r_[[cx, cy] - half, [cx, cy] + half]
        fill = cv2.contourArea(contour) / (length * width)  # a probe fills its rectangle
        conf = float(np.clip(fill, 0, 1) * min(1.0, length / width / 10))
        probes.append(Probe(xyxy, conf, ends, float(angle_from_keypoints(ends[0], ends[1])), False))
    return probes


def predictor(params, cache=None):
    def predict(image_path):
        if cache is None or image_path not in cache:
            image = cv2.imread(str(image_path))
            if cache is None:
                return detect(image, params), image.shape[:2]
            cache[image_path] = image
        image = cache[image_path]
        return detect(image, params), image.shape[:2]
    return predict


def tune(gt_dir, splits=("train", "val"), max_images=300, seed=0):
    """Grid search for the parameters with the best F1 on labelled frames."""
    from erc_pose.evaluate import score_predictions

    grid = dict(h_lo=[50, 58], h_hi=[82, 90], s_min=[80, 110, 140], v_min=[40, 80],
                min_len=[8, 14, 20], min_elong=[2.5, 4, 6])
    best, best_f1, cache = None, -1, {}
    for values in itertools.product(*grid.values()):
        params = Params(**dict(zip(grid, values)))
        tp = n_gt = n_pred = 0
        for split in splits:
            m = score_predictions(predictor(params, cache), gt_dir, split, max_images=max_images, seed=seed)
            n_gt, n_pred, tp = n_gt + m["n_gt"], n_pred + m["n_pred"], tp + m["n_matched"]
        f1 = 2 * tp / (n_gt + n_pred) if n_gt + n_pred else 0
        if f1 > best_f1:
            best, best_f1 = params, f1
            print(f"F1 {f1:.3f} {asdict(params)}", flush=True)
    return best


def main():
    from erc_pose.evaluate import score_predictions

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="pose dataset folder (images/<split>, labels/<split>)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--params", help="json file with tuned parameters (default: tune on train+val)")
    parser.add_argument("--extra-gt", nargs="*", default=[], help="more test sets to report, as name=folder")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.params:
        params = Params(**json.loads(Path(args.params).read_text()))
    else:
        params = tune(args.gt)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{args.name}_params.json").write_text(json.dumps(asdict(params), indent=2))

    sets = [(args.name, args.gt)] + [tuple(item.split("=", 1)) for item in args.extra_gt]
    for name, gt in sets:
        out = {"name": name, "task": "classic", "split": "test", "params_m": 0.0, **asdict(params)}
        out.update(score_predictions(predictor(params), gt, "test"))
        if name == args.name:
            images = list_images(Path(gt) / "images" / "test")[:100]
            frames = [cv2.imread(str(p)) for p in images]
            start = time.perf_counter()
            for frame in frames:
                detect(frame, params)
            out["fps_cpu"] = len(frames) / (time.perf_counter() - start)
        (Path(args.out) / f"{name}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
