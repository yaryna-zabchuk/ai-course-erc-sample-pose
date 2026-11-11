"""Run a model on images, video or a camera; prints one JSON line per frame.

    python -m erc_pose.infer --model pose --source /dev/arducam-mast --show
    python -m erc_pose.infer --weights models/obb.pt --source path/to/video.mp4 --save out

--model takes one or more names from erc_pose.models (pose, obb, unet, rcnn, classic, ...);
with several, n / p switch between them while --show is on.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from erc_pose.benchmark import IMAGE_SUFFIXES
from erc_pose.direction import apply_direction, load_direction_model
from erc_pose.models import MODELS, load_predictor
from erc_pose.orientation import wrap_angle


def draw(image, probes, arrows=False):
    """Draw boxes and axis lines on a copy of the image; arrows only when the direction is known."""
    image = image.copy()
    for probe in probes:
        x1, y1, x2, y2 = (int(round(v)) for v in probe.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{probe.conf:.2f}"
        if probe.angle_deg is not None:
            shown = probe.angle_deg if arrows else float(wrap_angle(probe.angle_deg, 180))
            label += f" {shown:.0f}deg"
            centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            half = np.hypot(x2 - x1, y2 - y1) / 2 * np.array(
                [np.cos(np.radians(probe.angle_deg)), np.sin(np.radians(probe.angle_deg))])
            start, end = (probe.kpts[0], probe.kpts[1]) if probe.kpts is not None else (centre - half, centre + half)
            start, end = tuple(int(v) for v in start), tuple(int(v) for v in end)
            if arrows and probe.directed:
                cv2.arrowedLine(image, start, end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
            else:
                cv2.line(image, start, end, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return image


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def camera_frames(device, width=None, height=None, rotate=0):
    """Frames from a camera index or a /dev path (udev aliases such as /dev/arducam-mast work too)."""
    if isinstance(device, str):
        if not Path(device).exists():
            present = sorted(str(p) for p in Path("/dev").glob("arducam*")) or ["none"]
            raise OSError(f"{device} does not exist; connected Arducam aliases: {', '.join(present)}")
        device = str(Path(device).resolve())  # /dev/arducam-mast -> /dev/video4
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise OSError(f"cannot open camera {device}")
    if width and height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {device}: {actual[0]}x{actual[1]}", file=sys.stderr, flush=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.rotate(frame, ROTATIONS[rotate]) if rotate else frame
    finally:
        capture.release()


def file_frames(source):
    """Frames of an image, a folder of images or a video file."""
    path = Path(source)
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        paths = [path]
    else:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"cannot open {source}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame
        finally:
            capture.release()
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            yield frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", nargs="+", choices=list(MODELS), help="model(s) from erc_pose.models")
    which.add_argument("--weights", help="Ultralytics weights or a spec, see erc_pose.models")
    parser.add_argument("--source", required=True, help="image, folder, video, camera index or /dev/... camera")
    parser.add_argument("--width", type=int, default=960, help="camera capture width (the training frames are 960)")
    parser.add_argument("--height", type=int, default=600, help="camera capture height (the training frames are 600)")
    parser.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                        help="rotate camera frames clockwise, e.g. 180 for a camera mounted upside down")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, help="default: the training image size")
    parser.add_argument("--device")
    parser.add_argument("--direction-weights",
                        help="pose model or direction classifier that gives OBB angles their direction")
    parser.add_argument("--direction-tta", action="store_true", help="flip test-time augmentation for the direction")
    parser.add_argument("--save", help="folder for annotated frames")
    parser.add_argument("--save-detections-only", action="store_true", help="with --save: skip frames with no probe")
    parser.add_argument("--save-raw", help="folder for the unannotated frames (for labelling): every --raw-every-th "
                        "frame, with or without a probe (missed probes are the ones worth labelling)")
    parser.add_argument("--raw-every", type=int, default=10, help="with --save-raw: keep every N-th frame")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--tile", action="store_true",
                        help="run every --model on each frame and show them side by side (instead of n/p switching)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(source, int) or args.source.startswith("/dev/")
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.save_raw) if args.save_raw else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
    session = time.strftime("%Y%m%d-%H%M%S")
    names = args.model or [args.weights]
    predictors = [load_predictor(n, args.device, args.conf, args.imgsz) for n in names]
    active = 0
    direction_model = load_direction_model(args.direction_weights) if args.direction_weights else None
    frames = camera_frames(source, args.width, args.height, args.rotate) if is_camera else file_frames(source)
    fps, last = None, time.perf_counter()
    for index, frame in enumerate(frames):
        if raw_dir and index % args.raw_every == 0:
            # session prefix keeps one run in one split
            cv2.imwrite(str(raw_dir / f"live{session}_{index:06d}.jpg"), frame)
        which = range(len(names)) if args.tile else [active]
        tiles = []
        for i in which:
            name = Path(names[i]).stem
            probes = predictors[i](frame)
            if direction_model is not None:
                probes = apply_direction(direction_model, frame, probes, args.conf, args.device, args.direction_tta)
            print(json.dumps({"frame": index, "model": name, "probes": [p.to_message() for p in probes]}), flush=True)
            if save_dir or args.show:
                image = draw(frame, probes, arrows=direction_model is not None)
                if save_dir and (probes or not args.save_detections_only):
                    cv2.imwrite(str(save_dir / f"{index:06d}_{name}.jpg"), image)
                tiles.append((name, len(probes), image))
        now = time.perf_counter()
        fps = 1 / (now - last) if fps is None else 0.9 * fps + 0.1 / (now - last)  # smoothed
        last = now
        if args.show:
            for name, count, image in tiles:
                cv2.putText(image, f"{name}  {fps:.1f} fps  {count} probe(s)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("probes (Esc or q to quit)", tile_images([t[2] for t in tiles]))
            key = cv2.waitKey(1)
            if key in (27, ord("q")):
                break
            if key in (ord("n"), ord("p")) and not args.tile:
                active = (active + (1 if key == ord("n") else -1)) % len(names)
                fps = None


def tile_images(images, max_width=1900):
    """Images side by side (two rows for four or more), scaled to fit the screen width."""
    if len(images) == 1:
        return images[0]
    columns = len(images) if len(images) <= 3 else (len(images) + 1) // 2
    h, w = images[0].shape[:2]
    scale = min(1.0, max_width / (columns * w))
    size = (int(w * scale), int(h * scale))
    images = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images]
    images += [np.zeros_like(images[0])] * (-len(images) % columns)
    rows = [np.hstack(images[r:r + columns]) for r in range(0, len(images), columns)]
    return np.vstack(rows)

if __name__ == "__main__":
    main()
