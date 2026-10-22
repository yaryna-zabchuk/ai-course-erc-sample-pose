"""Convert CVAT / Roboflow annotations to YOLO datasets.

Input is a COCO Keypoints JSON or an Ultralytics YOLO Pose export (boxes + 2
keypoints per probe; keypoint 0 = top, keypoint 1 = bottom).

One annotation source produces up to three datasets that share the same
split: `detect` (boxes), `pose` (boxes + 2 keypoints) and `obb` (oriented
boxes derived from the keypoints). Without keypoints only `detect` is built.

    python -m erc_pose.data.build_dataset --yolo-pose data/raw/cvat_export --out data/processed
    python -m erc_pose.data.build_dataset --coco data/raw/annotations.json --out data/processed
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from erc_pose.data.geometry import bbox_from_keypoints, is_keypoint_extent, obb_from_keypoints, xywh_to_corners
from erc_pose.data.split import DEFAULT_GROUP_REGEX, DEFAULT_SESSION_GAP_S, SPLITS, split_groups

CLASS_NAMES = {0: "probe"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# probe width / length, used where the box has to be rebuilt from the keypoints
DEFAULT_WIDTH_RATIO = 0.18
# OKS sigma per keypoint, used by both the pose loss and pose mAP. 0.5 is the
# Ultralytics default for a 2-point skeleton. Smaller values were tried and are
# worse: the loss 1 - exp(-e) saturates on small probes and stops giving a
# gradient (0.15 and 0.05 both gave worse keypoints and angles on the test split).
DEFAULT_KPT_SIGMA = 0.5
KEYPOINT_NAMES = ("top", "bottom")


def _fmt(values):
    return " ".join(f"{v:.6f}" for v in values)


def _keypoints(ann):
    """Annotated keypoints as (2, 3) [x, y, visibility], or None."""
    kpts = ann.get("keypoints")
    if not kpts or len(kpts) != 3 * len(KEYPOINT_NAMES):
        return None
    return np.asarray(kpts, dtype=float).reshape(-1, 3)


def order_left_to_right(kpts):
    """Put the left end first (the upper end for a vertical probe), whatever the label said."""
    (x1, y1), (x2, y2) = kpts[0, :2], kpts[1, :2]
    return kpts[::-1].copy() if (x2, y2) < (x1, y1) else kpts  # angle then lies in (-90, 90]


def label_lines(anns, width, height, kpt_order="labelled"):
    """YOLO label lines of one image for each task."""
    size = np.array([width, height], dtype=float)
    lines = {"detect": [], "pose": [], "obb": []}
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 <= x1 or y2 <= y1:
            continue
        box = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
        lines["detect"].append(f"0 {_fmt(box)}")

        kpts = _keypoints(ann)
        if kpts is None:
            kpts = np.zeros((len(KEYPOINT_NAMES), 3))
        elif kpt_order == "left-to-right" and (kpts[:, 2] > 0).all():
            kpts = order_left_to_right(kpts)
        visible = kpts[:, 2] > 0
        xy = np.clip(kpts[:, :2] / size, 0.0, 1.0) * visible[:, None]
        pose = " ".join(f"{_fmt(point)} {int(v)}" for point, v in zip(xy, kpts[:, 2]))
        lines["pose"].append(f"0 {_fmt(box)} {pose}")

        if visible.all() and not np.allclose(kpts[0, :2], kpts[1, :2]):
            corners = obb_from_keypoints(kpts[0, :2], kpts[1, :2], (w, h))
        else:
            corners = xywh_to_corners((x, y, w, h))
        lines["obb"].append(f"0 {_fmt(np.clip(corners / size, 0.0, 1.0).ravel())}")
    return lines


def _place(src, dst, copy):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def load_coco(coco_path, images_dir=None):
    """Images {stem: {path, width, height}} and annotations {stem: [COCO annotation]}."""
    coco_path = Path(coco_path)
    coco = json.loads(coco_path.read_text())
    root = Path(images_dir) if images_dir else coco_path.parent
    images, anns, ids = {}, defaultdict(list), {}
    for img in coco["images"]:
        stem = Path(img["file_name"]).stem
        images[stem] = {"path": root / img["file_name"], "width": img["width"], "height": img["height"]}
        ids[img["id"]] = stem
    for ann in coco["annotations"]:
        if not ann.get("iscrowd") and ann["image_id"] in ids:
            anns[ids[ann["image_id"]]].append(ann)
    return images, anns


def load_yolo_pose(export_dir, images_dir=None):
    """Same as load_coco, for an Ultralytics YOLO Pose export (CVAT "Ultralytics YOLO Pose 1.0").

    Labels are `cls cx cy w h x1 y1 [v1] x2 y2 [v2]`, normalised. The export's own
    subsets are ignored, since build() re-splits by recording session.
    """
    from PIL import Image

    export_dir = Path(export_dir)
    search = [export_dir / "images", export_dir] + ([Path(images_dir)] if images_dir else [])
    found = {}
    for folder in search:
        for path in sorted(folder.rglob("*")) if folder.exists() else []:
            if path.suffix.lower() in IMAGE_SUFFIXES:
                found.setdefault(path.stem, path)

    images, anns = {}, defaultdict(list)
    for label in sorted((export_dir / "labels").rglob("*.txt")):
        if label.stem not in found:
            raise FileNotFoundError(f"no image for {label}; export with images or pass --images")
        with Image.open(found[label.stem]) as image:
            width, height = image.size
        images[label.stem] = {"path": found[label.stem], "width": width, "height": height}
        for line in label.read_text().splitlines():
            values = [float(v) for v in line.split()]
            if len(values) < 5:
                continue
            cx, cy, w, h = values[1:5]
            ann = {"bbox": [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]}
            rest = values[5:]
            dims = 3 if len(rest) == 3 * len(KEYPOINT_NAMES) else 2 if len(rest) == 2 * len(KEYPOINT_NAMES) else 0
            if dims:
                kpts = np.asarray(rest).reshape(-1, dims)
                vis = kpts[:, 2] if dims == 3 else np.where(kpts[:, :2].any(1), 2.0, 0.0)
                ann["keypoints"] = [v for (x, y), visible in zip(kpts[:, :2], vis)
                                    for v in (x * width, y * height, visible)]
            anns[label.stem].append(ann)
    return images, anns


def repair_line_boxes(anns_by_image, width_ratio):
    """Give axis-line labels (box = extent of the two keypoints) a box with an area to train on.

    Returns the count.
    """
    repaired = 0
    for anns in anns_by_image.values():
        for ann in anns:
            kpts = _keypoints(ann)
            if kpts is None or not (kpts[:, 2] > 0).all() or np.allclose(kpts[0, :2], kpts[1, :2]):
                continue
            if is_keypoint_extent(ann["bbox"], kpts[0, :2], kpts[1, :2]):
                ann["bbox"] = bbox_from_keypoints(kpts[0, :2], kpts[1, :2], width_ratio)
                repaired += 1
    return repaired


def load_box_labels(export_dir):
    """Normalised (cx, cy, w, h) boxes per image stem from a YOLO Detection export (all classes)."""
    boxes = defaultdict(list)
    for label in sorted(Path(export_dir).glob("labels/**/*.txt")):
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) >= 5:
                boxes[label.stem].append([float(v) for v in values[1:5]])
    return boxes


def attach_boxes(images, anns_by_image, boxes):
    """Give each axis-line annotation the drawn box around it.

    A line belongs to the smallest box that contains both of its ends (3 px of
    slack), otherwise to the box whose centre is nearest the line's midpoint.
    Drawn boxes without a line are added as box-only annotations. Returns the
    number of lines that got a drawn box.
    """
    attached = 0
    for stem, normalised in boxes.items():
        if stem not in images:
            continue
        width, height = images[stem]["width"], images[stem]["height"]
        free = [[(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height] for cx, cy, w, h in normalised]
        for ann in anns_by_image[stem]:
            kpts = _keypoints(ann)
            if kpts is None or not free:
                continue
            ends = kpts[:, :2]

            def contains(box, slack=3.0):
                x, y, w, h = box
                return all(x - slack <= px <= x + w + slack and y - slack <= py <= y + h + slack for px, py in ends)

            inside = [i for i, box in enumerate(free) if contains(box)]
            if inside:
                best = min(inside, key=lambda i: free[i][2] * free[i][3])
            else:
                mid = ends.mean(0)
                best = min(range(len(free)), key=lambda i: np.hypot(free[i][0] + free[i][2] / 2 - mid[0],
                                                                     free[i][1] + free[i][3] / 2 - mid[1]))
            ann["bbox"] = free.pop(best)
            attached += 1
        anns_by_image[stem].extend({"bbox": box} for box in free)
    return attached


def build(coco_paths=(), out_dir="data/processed", images_dir=None, ratios=(0.70, 0.15, 0.15), seed=0,
          group_regex=DEFAULT_GROUP_REGEX, copy=False, yolo_dirs=(), width_ratio=DEFAULT_WIDTH_RATIO,
          session_gap=DEFAULT_SESSION_GAP_S, extra_train=(), kpt_sigma=DEFAULT_KPT_SIGMA, fold=None, folds=5,
          kpt_order="labelled", box_dirs=()):
    """Build the datasets and return {task: path to its data.yaml}.

    `extra_train` YOLO Pose folders (e.g. pseudo-labels) go into the train split
    only and do not change how the other images are split. `box_dirs` are YOLO
    Detection exports with drawn boxes for the same images; each axis line gets
    the box around it instead of an estimated one, and the OBB width comes from it.
    """
    out_dir = Path(out_dir)
    images, anns_by_image = {}, defaultdict(list)
    sources = [load_coco(path, images_dir) for path in coco_paths]
    sources += [load_yolo_pose(path, images_dir) for path in yolo_dirs]
    extra = [load_yolo_pose(path) for path in extra_train]
    if not sources:
        raise ValueError("pass at least one COCO file or YOLO Pose export")
    for source_images, source_anns in sources + extra:
        duplicates = images.keys() & source_images.keys()
        if duplicates:
            raise ValueError(f"duplicate image name: {sorted(duplicates)[0]}")
        images.update(source_images)
        anns_by_image.update(source_anns)
    for box_dir in box_dirs:
        attached = attach_boxes(images, anns_by_image, load_box_labels(box_dir))
        print(f"{attached} axis label(s) got a drawn box from {box_dir}")
    repaired = repair_line_boxes(anns_by_image, width_ratio)
    if repaired:
        print(f"{repaired} axis label(s) wrapped in a box of width {width_ratio} x length for training")
    has_keypoints = any(
        (kpts := _keypoints(ann)) is not None and bool((kpts[:, 2] > 0).all())
        for anns in anns_by_image.values() for ann in anns
    )

    missing = [str(info["path"]) for info in images.values() if not info["path"].exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) not found, e.g. {missing[0]}; pass --images")

    tasks = ("detect", "pose", "obb") if has_keypoints else ("detect",)
    extra_stems = {stem for source_images, _ in extra for stem in source_images}
    assignment = split_groups([s for s in images if s not in extra_stems], ratios, seed, group_regex, session_gap,
                              fold, folds)
    assignment.update((stem, "train") for stem in extra_stems)
    for task in tasks:
        for split in SPLITS:
            for kind in ("images", "labels"):
                folder = out_dir / task / kind / split
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True)

    for stem, info in images.items():
        split = assignment[stem]
        lines = label_lines(anns_by_image[stem], info["width"], info["height"], kpt_order)
        for task in tasks:
            _place(info["path"], out_dir / task / "images" / split / info["path"].name, copy)
            (out_dir / task / "labels" / split / f"{stem}.txt").write_text("\n".join(lines[task]) + "\n" if lines[task] else "")

    yamls = {}
    for task in tasks:
        # no `path` key: train.py passes an absolute yaml path and the dataset root defaults to its folder
        config = {split: f"images/{split}" for split in SPLITS}
        config["names"] = CLASS_NAMES
        if task == "pose":
            config["kpt_shape"] = [len(KEYPOINT_NAMES), 3]
            config["flip_idx"] = list(range(len(KEYPOINT_NAMES)))  # top/bottom do not swap under a flip
            config["kpt_oks_sigmas"] = [kpt_sigma] * len(KEYPOINT_NAMES)
        yamls[task] = out_dir / task / "data.yaml"
        yamls[task].write_text(yaml.safe_dump(config, sort_keys=False))

    counts = Counter(assignment.values())
    print(f"built {', '.join(tasks)} in {out_dir}: " + ", ".join(f"{s}={counts[s]}" for s in SPLITS)
          + (f" (train includes {len(extra_stems)} extra)" if extra_stems else ""))
    if not has_keypoints:
        print("no keypoint annotations found, so pose and obb datasets were skipped")
    return yamls


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("annotations (at least one)")
    source.add_argument("--coco", nargs="+", default=[], help="COCO (Keypoints) annotation JSON file(s)")
    source.add_argument("--yolo-pose", nargs="+", default=[], help="unzipped Ultralytics YOLO Pose export folder(s)")
    parser.add_argument("--extra-train", nargs="+", default=[],
                        help="YOLO Pose folder(s), e.g. pseudo-labels, added to the train split only")
    parser.add_argument("--boxes", nargs="+", default=[],
                        help="YOLO Detection export(s) with drawn boxes for the same images")
    parser.add_argument("--images", help="image folder (default: next to the annotations)")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                        help="regex whose first group is the recording session of a file stem")
    parser.add_argument("--session-gap", type=float, default=DEFAULT_SESSION_GAP_S,
                        help="for timestamped frames: seconds between frames that start a new session")
    parser.add_argument("--width-ratio", type=float, default=DEFAULT_WIDTH_RATIO,
                        help="probe width / length, for boxes rebuilt from keypoints")
    parser.add_argument("--kpt-sigma", type=float, default=DEFAULT_KPT_SIGMA,
                        help="OKS sigma of each keypoint (pose loss and pose mAP)")
    parser.add_argument("--kpt-order", choices=("labelled", "left-to-right"), default="labelled",
                        help="left-to-right: the left end is always keypoint 0 (drops the top/bottom meaning)")
    parser.add_argument("--fold", type=int, help="cross-validation: use session fold N as the test split")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    args = parser.parse_args()
    if not args.coco and not args.yolo_pose:
        parser.error("pass --coco or --yolo-pose")
    build(args.coco, args.out, args.images, tuple(args.ratios), args.seed, args.group_regex, args.copy,
          args.yolo_pose, args.width_ratio, args.session_gap, args.extra_train,
          args.kpt_sigma, args.fold, args.folds, args.kpt_order, args.boxes)


if __name__ == "__main__":
    main()
