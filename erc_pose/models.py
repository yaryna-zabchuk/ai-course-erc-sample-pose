"""Every model of the comparison behind one interface: frame (BGR) -> list of Probe.

    python -m erc_pose.models                                   # list the models and their weights
    python -m erc_pose.models --check data/processed_v2/pose/images/test   # load each, measure FPS

A model is given by name (see MODELS) or by spec:
    path/to/yolo.pt                        any Ultralytics detect / pose / obb / seg model
    classic[:params.json]  lines[:params.json]
    rcnn:weights.pt  unet:weights.pt  twostage:detector.pt,regressor.pt
"""

import argparse
import json
import time
from pathlib import Path

MODELS = {  # name: (spec, what it is)
    "pose": ("models/pose_v2.pt", "YOLOv8n-pose, two end points (main)"),
    "obb": ("models/obb_v2.pt", "YOLOv8n-OBB, rotated box (most precise axis)"),
    "pose11": ("models/pose11_v2.pt", "YOLO11n-pose"),
    "pose11s": ("models/pose11s_v2.pt", "YOLO11s-pose (bigger, best recall)"),
    "obb11": ("models/obb11_v2.pt", "YOLO11n-OBB"),
    "seg": ("models/seg_v2.pt", "YOLOv8n-seg, axis from the mask"),
    "seg11": ("models/seg11_v2.pt", "YOLO11n-seg, axis from the mask"),
    "detect": ("models/detect_v2.pt", "YOLOv8n box detector (no angle)"),
    "baseline": ("models/baseline_v2.pt", "YOLOv8n box detector, no augmentation (no angle)"),
    "rcnn": ("rcnn:models/rcnn_v2.pt", "Keypoint R-CNN, ResNet-50 FPN"),
    "unet": ("unet:models/unet_v2.pt", "U-Net probe pixels, axis from PCA"),
    "twostage": ("twostage:models/detect_v2.pt,models/angle_v2.pt", "YOLO boxes + ResNet18 angle per crop"),
    "classic": ("classic:models/classic_params.json", "green threshold + rotated rectangle, no learning"),
    "lines": ("lines:models/lines_params.json", "LSD line segments on green pixels, no learning"),
}


def default_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_predictor(spec, device=None, conf=0.25, imgsz=None):
    """Callable frame -> list of Probe for a model name or spec."""
    spec = MODELS.get(spec, (spec,))[0]
    kind, _, arg = spec.partition(":") if ":" in spec and not Path(spec).exists() else ("yolo", "", spec)
    device = device or default_device()

    if kind in ("classic", "lines"):
        import cv2

        module = __import__(f"erc_pose.{kind}", fromlist=["detect", "Params"])
        params = module.Params(**json.loads(Path(arg).read_text())) if arg and Path(arg).exists() else module.Params()
        if kind == "lines":
            lsd = cv2.createLineSegmentDetector()
            return lambda frame: module.detect(frame, params, lsd)
        return lambda frame: module.detect(frame, params)

    if kind == "rcnn":
        from erc_pose import rcnn

        model = rcnn.load(arg, device)
        return lambda frame: rcnn.predict(model, frame, device, conf)

    if kind == "unet":
        from erc_pose import semseg

        model = semseg.load(arg, device)
        return lambda frame: semseg.predict(model, frame, device)

    if kind == "twostage":
        from erc_pose.twostage import TwoStage

        detector, regressor = arg.split(",")
        return TwoStage(detector, regressor, device, conf)

    from ultralytics import YOLO

    from erc_pose.benchmark import train_imgsz
    from erc_pose.predict import probes_from_result

    model = YOLO(arg)
    size = imgsz or train_imgsz(model)
    return lambda frame: probes_from_result(model.predict(frame, conf=conf, imgsz=size, device=device,
                                                         verbose=False)[0])


def check(images, names, device=None, count=50, out=None):
    """Load every model and time it end to end on the same frames; optionally save {name: fps} as JSON."""
    import cv2

    from erc_pose.benchmark import list_images

    frames = [cv2.imread(str(p)) for p in list_images(images)[:count]]
    speeds = {}
    for name in names:
        try:
            predict = load_predictor(name, device)
            for frame in frames[:3]:
                predict(frame)
            start, found = time.perf_counter(), 0
            for frame in frames:
                found += len(predict(frame))
            fps = len(frames) / (time.perf_counter() - start)
            speeds[name] = fps
            print(f"{name:10s} ok    {fps:6.1f} FPS  {found} probes in {len(frames)} frames", flush=True)
        except Exception as error:  # report and go on with the others
            print(f"{name:10s} FAIL  {type(error).__name__}: {error}", flush=True)
    if out:
        Path(out).write_text(json.dumps({"device": device or default_device(), "frames": len(frames),
                                         "fps": speeds}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="IMAGES", help="folder of frames: load every model and measure FPS")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--out", help="with --check: save the FPS of every model to this JSON file")
    args = parser.parse_args()
    if args.check:
        check(args.check, args.models, args.device, out=args.out)
        return
    for name, (spec, about) in MODELS.items():
        files = spec.split(":", 1)[-1].split(",")
        ready = all(Path(f).exists() for f in files)
        print(f"{name:10s} {'ready  ' if ready else 'MISSING'} {about}  [{spec}]")


if __name__ == "__main__":
    main()
