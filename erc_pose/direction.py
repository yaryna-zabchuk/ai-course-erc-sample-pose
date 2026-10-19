"""Which end of a probe is the top: a small classifier on axis-aligned crops.

Each probe is cut out along its axis into a square crop with one end on the
left. The classifier says whether the top (thick end) is on the left or on
the right. Every labelled probe gives two training crops, as-is and mirrored.

    python -m erc_pose.direction build --dataset data/processed/pose --out data/direction
    python -m erc_pose.train configs/direction.yaml
    python -m erc_pose.evaluate --weights runs/obb/weights/best.pt --data data/processed/obb/data.yaml \
        --direction-weights runs/direction/weights/best.pt --name obb+direction
"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from erc_pose.data.split import SPLITS
from erc_pose.orientation import wrap_angle

CROP_SIZE = 64
LEFT, RIGHT = "top_left", "top_right"  # class names, sorted: index 0 = top_left


def canonical_crop(image, start, end, size=CROP_SIZE, margin=0.15):
    """Square crop with the probe axis horizontal, `start` on the left and `end` on the right."""
    start, end = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
    length = max(float(np.linalg.norm(end - start)), 1.0)
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    scale = size * (1 - 2 * margin) / length
    centre = (start + end) / 2
    # destination = scale * R * (source - centre) + size / 2, with R mapping axis -> +x
    rotation = np.stack([axis, normal]) * scale
    offset = np.full(2, size / 2, dtype=np.float32) - rotation @ centre
    return cv2.warpAffine(image, np.hstack([rotation, offset[:, None]]), (size, size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build(dataset_dir, out_dir):
    """Classification dataset from a pose dataset: out/{train,val,test}/{top_left,top_right}/*.png."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {}
    for split in SPLITS:
        for name in (LEFT, RIGHT):
            (out_dir / split / name).mkdir(parents=True)
        n = 0
        for image_path in sorted((dataset_dir / "images" / split).iterdir()):
            label = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            if not label.exists():
                continue
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            for index, line in enumerate(label.read_text().splitlines()):
                values = np.array(line.split(), dtype=float)
                if len(values) != 11 or values[7] <= 0 or values[10] <= 0:
                    continue
                top = values[5:7] * [width, height]
                bottom = values[8:10] * [width, height]
                if np.linalg.norm(bottom - top) < 4:
                    continue
                stem = f"{image_path.stem}_{index}"
                cv2.imwrite(str(out_dir / split / LEFT / f"{stem}.png"), canonical_crop(image, top, bottom))
                cv2.imwrite(str(out_dir / split / RIGHT / f"{stem}.png"), canonical_crop(image, bottom, top))
                n += 1
        counts[split] = n
    print(f"direction crops in {out_dir}: " + ", ".join(f"{s}={2 * n}" for s, n in counts.items()))
    return out_dir


def orient_probes(classifier, image, probes):
    """Give undirected probes with axis ends a direction from the classifier."""
    todo = [i for i, p in enumerate(probes) if p.kpts is not None and not p.directed]
    if not todo:
        return list(probes)
    crops = [canonical_crop(image, probes[i].kpts[0], probes[i].kpts[1]) for i in todo]
    results = classifier.predict(crops, imgsz=CROP_SIZE, verbose=False)
    out = list(probes)
    for i, result in zip(todo, results):
        probe = probes[i]
        p_left = float(result.probs.data[classifier.names_index[LEFT]])
        angle, kpts = probe.angle_deg, probe.kpts
        if p_left < 0.5:  # top is the second axis end
            angle, kpts = float(wrap_angle(angle + 180)), kpts[::-1].copy()
        out[i] = replace(probe, angle_deg=float(angle), kpts=kpts, directed=True,
                         direction_conf=abs(p_left - 0.5) * 2)
    return out


def load_direction_model(weights):
    """A pose model or a direction classifier, either of which can direct OBB probes."""
    from ultralytics import YOLO

    model = YOLO(weights)
    if model.task == "classify":
        model.names_index = {name: index for index, name in model.names.items()}
    elif model.task != "pose":
        raise ValueError(f"{weights} is a {model.task} model; use a pose model or a direction classifier")
    return model


def apply_direction(model, image, probes, conf=0.25, device=None, tta=False):
    """Direct OBB probes with `model`, a pose model or a direction classifier.

    With `tta`, a pose model also looks at the horizontally, vertically and
    doubly flipped image and all its detections vote on the direction.
    """
    from erc_pose.benchmark import train_imgsz
    from erc_pose.predict import fuse_direction, probes_from_result, unflip

    if model.task == "classify":
        return orient_probes(model, image, probes)
    height, width = image.shape[:2]
    votes = []
    for flip_code in (None, 1, 0, -1) if tta else (None,):
        view = image if flip_code is None else cv2.flip(image, flip_code)
        result = model.predict(view, imgsz=train_imgsz(model), conf=conf, device=device, verbose=False)[0]
        found = probes_from_result(result)
        votes += found if flip_code is None else [unflip(p, flip_code, width, height) for p in found]
    return fuse_direction(probes, votes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="make the crop dataset from a pose dataset")
    b.add_argument("--dataset", default="data/processed/pose")
    b.add_argument("--out", default="data/direction")
    args = parser.parse_args()
    if args.command == "build":
        build(args.dataset, args.out)


if __name__ == "__main__":
    main()
