"""Pre-label frames with a trained model and pack them for CVAT.

    python -m erc_pose.data.prelabel --weights runs/obb_semi/weights/best.pt \
        --images data/raw/erc-sample-pose/data --out data/cvat_prelabels_obb
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

from erc_pose.benchmark import IMAGE_SUFFIXES, train_imgsz
from erc_pose.predict import probes_from_result


# label names in our CVAT task
POSE_NAMES = ["sample_pose"]
BOX_NAMES = ["sample_box", "sample_pose"]


def prelabel(weights, image_dirs, labelled_dirs, out_dir, conf=0.25, with_images=True, names=None):
    """Write pre-labels as a zip CVAT can import."""
    from ultralytics import YOLO

    model = YOLO(weights)
    if model.task not in ("obb", "pose", "detect"):
        raise ValueError(f"{weights} is a {model.task} model; pre-labels need an OBB, pose or detection model")
    boxes_only = model.task == "detect"
    names = names or (BOX_NAMES if boxes_only else POSE_NAMES)
    labelled = {p.stem for d in labelled_dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    images = sorted({p.stem: p for d in image_dirs for p in Path(d).rglob("*")
                     if p.suffix.lower() in IMAGE_SUFFIXES and p.stem not in labelled}.values())

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "labels" / "train").mkdir(parents=True)
    if with_images:
        (out_dir / "images" / "train").mkdir(parents=True)
    n_probes = 0
    for path in images:
        result = model.predict(str(path), imgsz=train_imgsz(model), conf=conf, verbose=False)[0]
        height, width = result.orig_shape
        lines = []
        for probe in probes_from_result(result):
            if boxes_only:
                x1, y1, x2, y2 = np.clip(probe.xyxy / [width, height, width, height], 0, 1)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)))
                continue
            points = np.clip(probe.kpts / [width, height], 0, 1)
            (x1, y1), (x2, y2) = points.min(0), points.max(0)
            box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]  # line extent, like CVAT exports
            lines.append("0 " + " ".join(f"{v:.6f}" for v in box)
                         + f" {points[0, 0]:.6f} {points[0, 1]:.6f} 2 {points[1, 0]:.6f} {points[1, 1]:.6f} 2")
        n_probes += len(lines)
        (out_dir / "labels" / "train" / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if with_images:
            shutil.copy2(path, out_dir / "images" / "train" / path.name)
    config = {"path": ".", "train": "images/train", "names": dict(enumerate(names))}
    if not boxes_only:
        config.update(kpt_shape=[2, 3], flip_idx=[0, 1])
    (out_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    kind = "boxes" if boxes_only else "axis lines"
    print(f"{len(images)} frames, {n_probes} pre-labelled probes ({kind}) -> {archive}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="OBB or pose model (axis lines) or detection model (boxes)")
    parser.add_argument("--images", nargs="+", required=True, help="folders with the frames to pre-label")
    parser.add_argument("--labelled", nargs="*", default=[], help="folders whose frames are already labelled")
    parser.add_argument("--out", default="data/cvat_prelabels")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-images", action="store_true", help="labels only, for a CVAT task that has the frames")
    parser.add_argument("--names", nargs="+", help="label names in the CVAT task, class 0 first "
                        f"(default: {' '.join(POSE_NAMES)} for axis lines, {' '.join(BOX_NAMES)} for boxes)")
    args = parser.parse_args()
    prelabel(args.weights, args.images, args.labelled, args.out, args.conf, not args.no_images, args.names)


if __name__ == "__main__":
    main()
