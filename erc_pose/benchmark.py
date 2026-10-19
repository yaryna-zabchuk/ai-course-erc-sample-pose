"""Inference speed of a trained model.

    python -m erc_pose.benchmark --weights runs/pose/weights/best.pt --images data/processed/pose/images/test
"""

import argparse
import time
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def measure_fps(model, images, device, imgsz=640, warmup=5, max_images=100, direction_model=None, tta=False):
    """End-to-end FPS (pre-process + inference + post-process), one image at a time.

    With `direction_model`, the time includes giving the OBB probes their direction.
    """
    from erc_pose.direction import apply_direction
    from erc_pose.predict import probes_from_result

    frames = [cv2.imread(str(p)) for p in images[:max_images]]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("no readable images to benchmark on")

    def run(frame):
        result = model.predict(frame, device=device, imgsz=imgsz, verbose=False)[0]
        if direction_model is not None:
            apply_direction(direction_model, frame, probes_from_result(result), device=device, tta=tta)

    for frame in frames[:warmup]:
        run(frame)
    start = time.perf_counter()
    for frame in frames:
        run(frame)
    return len(frames) / (time.perf_counter() - start)


def train_imgsz(model, default=640):
    """Image size the model was trained at, so inference sees the same scale."""
    return (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", default)


def available_devices():
    import torch

    return ["cpu", 0] if torch.cuda.is_available() else ["cpu"]


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True, help="folder with images")
    parser.add_argument("--imgsz", type=int, help="default: the training image size")
    parser.add_argument("--direction-weights", help="include directing OBB probes with this model")
    args = parser.parse_args()
    model, images = YOLO(args.weights), list_images(args.images)
    direction_model = None
    if args.direction_weights:
        from erc_pose.direction import load_direction_model

        direction_model = load_direction_model(args.direction_weights)
    for device in available_devices():
        name = "cpu" if device == "cpu" else "gpu"
        print(f"{name}: {measure_fps(model, images, device, args.imgsz or train_imgsz(model),
                                           direction_model=direction_model):.1f} FPS")


if __name__ == "__main__":
    main()
