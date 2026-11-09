"""Second classical baseline: straight line segments (LSD) that lie on green pixels.

    python -m erc_pose.lines --gt data/processed_v2/pose --name lines_v2

The green mask only says where to look; the angle comes from the line segments
along the probe edges (length-weighted mean of the doubled angles).
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
from erc_pose.predict import Probe


@dataclass
class Params:
    h_lo: int = 58        # green mask, as tuned for erc_pose.classic
    h_hi: int = 82
    s_min: int = 140
    v_min: int = 40
    min_seg: float = 6    # px, shortest line segment kept
    on_green: float = 0.6  # share of a segment's points that must be green
    min_len: float = 15   # px, shortest probe
    min_conf: float = 0.0  # segment agreement x coverage


def box_around(ends, width_ratio=DEFAULT_WIDTH_RATIO):
    """Axis-aligned box of a width_ratio x length rectangle around the axis, as for the training boxes."""
    centre, d = ends.mean(0), ends[1] - ends[0]
    length = max(np.linalg.norm(d), 1e-6)
    half = np.abs(d) / 2 + np.abs(d[::-1]) / length * width_ratio * length / 2
    return np.r_[centre - half, centre + half]


def prepare(image, p=Params(), lsd=None):
    """Green mask, its components and the line segments; independent of the filter parameters."""
    lsd = lsd or cv2.createLineSegmentDetector()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (p.h_lo, p.s_min, p.v_min), (p.h_hi, 255, 255))
    near = cv2.dilate(green, np.ones((5, 5), np.uint8))  # edges lie just outside the green pixels
    labels = cv2.connectedComponents(near)[1]
    segments = lsd.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
    return green, near, labels, np.zeros((0, 4)) if segments is None else segments.reshape(-1, 4)


def detect(image, p=Params(), lsd=None, prepared=None):
    """Probes in a BGR image."""
    green, near, labels, segments = prepared or prepare(image, p, lsd)
    h, w = green.shape
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    segments, length = segments[length >= p.min_seg], length[length >= p.min_seg]
    t = np.linspace(0, 1, 9)
    xs = np.clip((segments[:, :1] + t * (segments[:, 2:3] - segments[:, :1])).round().astype(int), 0, w - 1)
    ys = np.clip((segments[:, 1:2] + t * (segments[:, 3:4] - segments[:, 1:2])).round().astype(int), 0, h - 1)
    keep = (near[ys, xs] > 0).mean(1) >= p.on_green
    # component of each kept segment: the one under most of its points (all of them, given on_green >= 0.5)
    component = np.array([np.bincount(row).argmax() for row in labels[ys[keep], xs[keep]]], dtype=int)
    groups = {}
    for c, seg, l in zip(component, segments[keep], length[keep]):
        if c:
            groups.setdefault(c, []).append((*seg, l))

    probes = []
    for component, segs in groups.items():
        segs = np.array(segs)
        theta = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
        weight = segs[:, 4]
        mean = 0.5 * np.arctan2((weight * np.sin(2 * theta)).sum(), (weight * np.cos(2 * theta)).sum())
        d = np.array([np.cos(mean), np.sin(mean)])
        ys, xs = np.nonzero((labels == component) & (green > 0))
        if len(xs) < 3:
            continue
        pts = np.c_[xs, ys].astype(float)
        centre = pts.mean(0)
        proj = (pts - centre) @ d
        ends = np.stack([centre + proj.min() * d, centre + proj.max() * d])
        length = proj.max() - proj.min()
        if length < p.min_len:
            continue
        # straightness: how much of the probe length the segments cover, and how well they agree
        agreement = float(np.abs((weight * np.exp(2j * theta)).sum()) / weight.sum())
        conf = float(np.clip(agreement * min(1.0, weight.sum() / (2 * length)), 0, 1))
        if conf < p.min_conf:
            continue
        probes.append(Probe(box_around(ends), conf, ends, float(angle_from_keypoints(ends[0], ends[1])), False))
    return probes


def predictor(params, cache=None):
    lsd = cv2.createLineSegmentDetector()

    def predict(image_path):
        if cache is None:
            image = cv2.imread(str(image_path))
            return detect(image, params, lsd), image.shape[:2]
        if image_path not in cache:
            image = cv2.imread(str(image_path))
            cache[image_path] = (prepare(image, params, lsd), image.shape[:2])
        prepared, shape = cache[image_path]
        return detect(None, params, prepared=prepared), shape
    return predict


def tune(gt_dir, splits=("train", "val"), max_images=300, seed=0):
    """Grid search for the parameters with the best F1 on labelled frames."""
    from erc_pose.evaluate import score_predictions

    grid = dict(min_seg=[4, 6, 10], on_green=[0.4, 0.6, 0.8], min_len=[10, 15, 20, 30],
                min_conf=[0.0, 0.2, 0.4, 0.6])
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.params:
        params = Params(**json.loads(Path(args.params).read_text()))
    else:
        params = tune(args.gt)
        (out_dir / f"{args.name}_params.json").write_text(json.dumps(asdict(params), indent=2))

    for name, gt in [(args.name, args.gt)] + [tuple(item.split("=", 1)) for item in args.extra_gt]:
        out = {"name": name, "task": "lines", "split": "test", "params_m": 0.0, **asdict(params)}
        out.update(score_predictions(predictor(params), gt, "test"))
        if name == args.name:
            frames = [cv2.imread(str(p)) for p in list_images(Path(gt) / "images" / "test")[:100]]
            lsd = cv2.createLineSegmentDetector()
            start = time.perf_counter()
            for frame in frames:
                detect(frame, params, lsd)
            out["fps_cpu"] = len(frames) / (time.perf_counter() - start)
        (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
