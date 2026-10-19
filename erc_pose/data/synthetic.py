"""Generate synthetic probe images with COCO-keypoints annotations.

Green cylinders with a dark top cap on a noisy reddish background. Useful for
smoke-testing the pipeline and as extra training data (Plan B in the proposal).

    python -m erc_pose.data.synthetic --out data/synthetic --count 300
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _background(rng, width, height):
    base = np.array([rng.integers(40, 90), rng.integers(60, 110), rng.integers(130, 200)], dtype=np.float32)
    coarse = rng.normal(0, 18, (height // 16 + 1, width // 16 + 1, 1)).astype(np.float32)
    texture = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)[..., None]
    image = base + texture + rng.normal(0, 6, (height, width, 3))
    for _ in range(rng.integers(0, 6)):  # rocks
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(8, 50)), int(rng.integers(8, 50)))
        cv2.ellipse(image, centre, axes, float(rng.uniform(0, 180)), 0, 360, (base * rng.uniform(0.5, 0.9)).tolist(), -1)
    return image


def _draw_probe(image, rng):
    height, width = image.shape[:2]
    length = rng.uniform(0.12, 0.45) * min(width, height)
    thickness = length * rng.uniform(0.10, 0.18)
    theta = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)])
    margin = length / 2 + thickness
    centre = np.array([rng.uniform(margin, width - margin), rng.uniform(margin, height - margin)])
    top, bottom = centre - direction * length / 2, centre + direction * length / 2

    green = np.array([rng.integers(20, 70), rng.integers(130, 230), rng.integers(20, 80)], dtype=float)
    normal = np.array([-direction[1], direction[0]]) * thickness / 2
    body = np.stack([top + normal, bottom + normal, bottom - normal, top - normal])
    cap_end = top + direction * length * 0.18
    cap = np.stack([top + normal, cap_end + normal, cap_end - normal, top - normal])
    cv2.fillConvexPoly(image, np.round(body).astype(np.int32), green.tolist(), cv2.LINE_AA)
    cv2.fillConvexPoly(image, np.round(cap).astype(np.int32), (green * 0.25).tolist(), cv2.LINE_AA)

    x1, y1 = body.min(0)
    x2, y2 = body.max(0)
    return {
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "keypoints": [float(top[0]), float(top[1]), 2, float(bottom[0]), float(bottom[1]), 2],
    }


def generate(out_dir, count=300, sessions=8, width=640, height=480, seed=0):
    """Write `count` images and `annotations.json` into out_dir; return the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "probe", "keypoints": ["top", "bottom"], "skeleton": [[1, 2]]}],
    }
    for index in range(count):
        image = _background(rng, width, height)
        name = f"synth{index % sessions:02d}_{index:05d}.jpg"
        for _ in range(rng.integers(1, 4)):
            ann = _draw_probe(image, rng)
            ann.update(id=len(coco["annotations"]) + 1, image_id=index, category_id=1, num_keypoints=2,
                       area=ann["bbox"][2] * ann["bbox"][3], iscrowd=0)
            coco["annotations"].append(ann)
        image = image * rng.uniform(0.6, 1.2)  # global lighting
        cv2.imwrite(str(out_dir / name), np.clip(image, 0, 255).astype(np.uint8))
        coco["images"].append({"id": index, "file_name": name, "width": width, "height": height})
    path = out_dir / "annotations.json"
    path.write_text(json.dumps(coco))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--sessions", type=int, default=8, help="number of fake recording sessions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(generate(args.out, args.count, args.sessions, seed=args.seed))


if __name__ == "__main__":
    main()
