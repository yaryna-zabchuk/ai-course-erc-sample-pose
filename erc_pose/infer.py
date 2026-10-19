"""Run a trained model on images, a video or a camera and print compact detections.

One JSON line per frame is all that has to go over the radio link:

    python -m erc_pose.infer --weights runs/obb_semi/weights/best.pt --source 0 --show
    python -m erc_pose.infer --weights runs/obb_semi/weights/best.pt --source frames/ --save out/
"""

import argparse
import json
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


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image, folder, video or camera index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, help="default: the training image size")
    parser.add_argument("--device")
    parser.add_argument("--direction-weights",
                        help="pose model or direction classifier that gives OBB angles their direction")
    parser.add_argument("--direction-tta", action="store_true", help="flip test-time augmentation for the direction")
    parser.add_argument("--save", help="folder for annotated frames")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    direction_model = load_direction_model(args.direction_weights) if args.direction_weights else None
    results = model.predict(source, stream=True, conf=args.conf, imgsz=args.imgsz or train_imgsz(model),
                            device=args.device, verbose=False)
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
        if save_dir or args.show:
            image = draw(result.orig_img, probes)
            if save_dir:
                cv2.imwrite(str(save_dir / f"{index:06d}.jpg"), image)
            if args.show:
                cv2.putText(image, f"{fps:.1f} fps  {len(probes)} probe(s)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("probes (Esc or q to quit)", image)
                if cv2.waitKey(1) in (27, ord("q")):
                    break


if __name__ == "__main__":
    main()
