"""Run a model on images, video or a camera; prints one JSON line per frame.

    python -m erc_pose.infer --weights models/pose.pt --source /dev/arducam-mast --show
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from erc_pose.benchmark import train_imgsz
from erc_pose.direction import apply_direction, load_direction_model
from erc_pose.predict import probes_from_result


def draw(image, probes):
    """Draw boxes and orientation arrows on a copy of the image."""
    image = image.copy()
    for probe in probes:
        x1, y1, x2, y2 = (int(round(v)) for v in probe.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{probe.conf:.2f}"
        if probe.angle_deg is not None:
            label += f" {probe.angle_deg:.0f}deg"
            centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            half = np.hypot(x2 - x1, y2 - y1) / 2 * np.array(
                [np.cos(np.radians(probe.angle_deg)), np.sin(np.radians(probe.angle_deg))])
            start, end = (probe.kpts[0], probe.kpts[1]) if probe.kpts is not None else (centre - half, centre + half)
            start, end = tuple(int(v) for v in start), tuple(int(v) for v in end)
            if probe.directed:
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


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
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
    parser.add_argument("--save-raw", help="folder for the unannotated frames (for labelling): every --raw-every-th frame, "
                        "only those with a probe when --save-detections-only is set")
    parser.add_argument("--raw-every", type=int, default=10, help="with --save-raw: keep every N-th frame")
    parser.add_argument("--show", action="store_true")
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
    model = YOLO(args.weights)
    direction_model = load_direction_model(args.direction_weights) if args.direction_weights else None
    imgsz = args.imgsz or train_imgsz(model)
    if is_camera:  # ultralytics' loader can't set resolution or open /dev paths
        results = (model.predict(frame, conf=args.conf, imgsz=imgsz, device=args.device, verbose=False)[0]
                   for frame in camera_frames(source, args.width, args.height, args.rotate))
    else:
        results = model.predict(source, stream=True, conf=args.conf, imgsz=imgsz, device=args.device, verbose=False)
    fps, last = None, time.perf_counter()
    for index, result in enumerate(results):
        now = time.perf_counter()
        fps = 1 / (now - last) if fps is None else 0.9 * fps + 0.1 / (now - last)  # smoothed
        last = now
        probes = probes_from_result(result)
        if direction_model is not None:
            probes = apply_direction(direction_model, result.orig_img, probes, args.conf, args.device,
                                     args.direction_tta)
        print(json.dumps({"frame": index, "probes": [p.to_message() for p in probes]}), flush=True)
        if raw_dir and index % args.raw_every == 0 and (probes or not args.save_detections_only):
            # session prefix keeps one run in one split
            cv2.imwrite(str(raw_dir / f"live{session}_{index:06d}.jpg"), result.orig_img)
        if save_dir or args.show:
            image = draw(result.orig_img, probes)
            if save_dir and (probes or not args.save_detections_only):
                cv2.imwrite(str(save_dir / f"{index:06d}.jpg"), image)
            if args.show:
                cv2.putText(image, f"{fps:.1f} fps  {len(probes)} probe(s)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("probes (Esc or q to quit)", image)
                if cv2.waitKey(1) in (27, ord("q")):
                    break


if __name__ == "__main__":
    main()
