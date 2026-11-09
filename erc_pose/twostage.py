"""Two stages: a box detector finds the probes, a small CNN gives the axis angle of each crop.

    python -m erc_pose.twostage train --data data/processed_v2/pose --out runs/angle_v2
    python -m erc_pose.twostage evaluate --detector runs/detect_v2/weights/best.pt \
        --weights runs/angle_v2/last.pt --gt data/processed_v2/pose --name twostage_v2

The regressor sees a square crop around the box (so resizing keeps the angle) and
predicts (cos 2θ, sin 2θ), which is the same for both ends of the probe. Unlike the
box models it can be trained with any rotation: the label rotates with the crop.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from erc_pose.benchmark import list_images
from erc_pose.evaluate import read_ground_truth, score_predictions
from erc_pose.orientation import angular_error
from erc_pose.predict import Probe
from erc_pose.semseg import to_tensor

CROP = 128        # regressor input, px
CONTEXT = 1.25    # crop side = CONTEXT x longer box side


def build_model(pretrained=True):
    from torchvision.models import resnet18

    model = resnet18(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def crop(image, centre, side, rotate_deg=0.0, size=CROP):
    """Square crop of `side` px around `centre`, rotated by rotate_deg (OpenCV sense), resized to size."""
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), rotate_deg, size / side)
    m[:, 2] += size / 2 - np.asarray(centre, dtype=float)
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def probe_crops(root, split):
    """(image path, centre, side, axis angle in degrees) for every labelled probe."""
    from PIL import Image

    items = []
    for path in list_images(Path(root) / "images" / split):
        with Image.open(path) as image:
            w, h = image.size
        boxes, kpts = read_ground_truth(Path(root) / "labels" / split / f"{path.stem}.txt", w, h)
        for box, k in zip(boxes, kpts):
            if np.isnan(k).any():
                continue
            d = k[1, :2] - k[0, :2]
            items.append((path, (box[:2] + box[2:]) / 2, max(box[2] - box[0], box[3] - box[1]) * CONTEXT,
                          float(np.degrees(np.arctan2(d[1], d[0])))))
    return items


class CropSet(torch.utils.data.Dataset):
    def __init__(self, root, split, train=False, repeats=1):
        self.items = probe_crops(root, split)
        self.train, self.repeats = train, repeats
        self.images = {}

    def __len__(self):
        return len(self.items) * self.repeats

    def image(self, path):
        if path not in self.images:
            self.images[path] = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.images[path]

    def __getitem__(self, i):
        path, centre, side, angle = self.items[i % len(self.items)]
        side = max(side, 16.0)
        rotate = 0.0
        if self.train:  # detector boxes are not exact: jitter the crop, then rotate freely
            centre = centre + np.random.uniform(-0.1, 0.1, 2) * side
            side = side * np.random.uniform(0.85, 1.25)
            rotate = np.random.uniform(0, 360)
        patch = crop(self.image(path), centre, side, rotate)
        angle = angle - rotate  # OpenCV turns the picture counter-clockwise on screen: y-down angles drop
        if self.train:
            if np.random.rand() < 0.5:
                patch, angle = patch[:, ::-1], 180 - angle
            patch = np.clip(patch * np.random.uniform(0.6, 1.4) + np.random.uniform(-25, 25), 0, 255)
        t = np.radians(2 * angle)
        return to_tensor(np.ascontiguousarray(patch)), torch.tensor([np.cos(t), np.sin(t)], dtype=torch.float32)


def to_angle(output):
    """Axis angle in degrees, (-90, 90], from (cos 2θ, sin 2θ)."""
    return np.degrees(np.arctan2(output[:, 1], output[:, 0])) / 2


def train(data, out, epochs=30, batch=64, lr=1e-3, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    train_set, val_set = CropSet(data, "train", train=True, repeats=4), CropSet(data, "val")
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=4, drop_last=True,
                                         worker_init_fn=lambda i: np.random.seed(torch.initial_seed() % 2**32))
    val = torch.utils.data.DataLoader(val_set, batch_size=batch, num_workers=2)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs * len(loader), pct_start=0.1)
    log = open(out / "train.log", "a")
    for epoch in range(epochs):
        model.train()
        total, start = 0.0, time.time()
        for x, y in loader:
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item()
        model.eval()
        pred, gt = [], []
        with torch.no_grad():
            for x, y in val:
                pred.append(model(x.to(device)).cpu().numpy())
                gt.append(y.numpy())
        error = angular_error(to_angle(np.concatenate(pred)), to_angle(np.concatenate(gt)), 180.0)
        line = (f"epoch {epoch + 1}/{epochs} loss {total / len(loader):.4f} val axis error {error.mean():.2f}° "
                f"(median {np.median(error):.2f}°) {time.time() - start:.0f}s")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        torch.save(model.state_dict(), out / "last.pt")
    return out / "last.pt"


class TwoStage:
    def __init__(self, detector, weights, device="cuda", conf=0.25):
        from ultralytics import YOLO

        self.detector, self.device, self.conf = YOLO(detector), device, conf
        self.imgsz = (getattr(self.detector, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 960)
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, image):
        """Probes in a BGR image."""
        result = self.detector.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        boxes, confs = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
        if not len(boxes):
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        sides = np.maximum((boxes[:, 2:] - boxes[:, :2]).max(1) * CONTEXT, 16)
        x = torch.stack([to_tensor(crop(rgb, c, s)) for c, s in zip(centres, sides)]).to(self.device)
        angles = to_angle(self.model(x).cpu().numpy())
        probes = []
        for box, conf, centre, angle in zip(boxes, confs, centres, angles):
            d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            w, h = box[2:] - box[:2]
            length = min(w / max(abs(d[0]), 1e-6), h / max(abs(d[1]), 1e-6), np.hypot(w, h))
            ends = np.stack([centre - d * length / 2, centre + d * length / 2])
            probes.append(Probe(box, float(conf), ends, float(angle), False))
        return probes


def evaluate(detector, weights, gt, name, extra=(), out_dir="results"):
    model = TwoStage(detector, weights)
    params_m = (sum(p.numel() for p in model.model.parameters())
                + sum(p.numel() for p in model.detector.model.parameters())) / 1e6
    for set_name, folder in [(name, gt), *extra]:
        def run(path):
            image = cv2.imread(str(path))
            return model(image), image.shape[:2]
        result = {"name": set_name, "task": "twostage", "detector": str(detector), "weights": str(weights),
                  "split": "test", "params_m": params_m, **score_predictions(run, folder, "test")}
        # the regressor alone, on crops around the labelled boxes
        crops = CropSet(folder, "test")
        with torch.no_grad():
            pred = np.concatenate([model.model(torch.stack([crops[j][0] for j in range(i, min(i + 64, len(crops)))])
                                               .to(model.device)).cpu().numpy() for i in range(0, len(crops), 64)])
        error = angular_error(to_angle(pred), np.array([a for *_, a in crops.items]), 180.0)
        result.update(regressor_on_labelled_boxes_mae_deg=float(error.mean()),
                      regressor_on_labelled_boxes_within_5deg=float((error <= 5).mean()))
        if set_name == name:
            frames = [cv2.imread(str(p)) for p in list_images(Path(folder) / "images" / "test")[:100]]
            for dev, count in (("cuda", 100), ("cpu", 30)):
                m = TwoStage(detector, weights, dev)
                for frame in frames[:3]:
                    m(frame)
                start = time.perf_counter()
                for frame in frames[:count]:
                    m(frame)
                result["fps_gpu" if dev == "cuda" else "fps_cpu"] = count / (time.perf_counter() - start)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{set_name}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="pose dataset folder")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    e = sub.add_parser("evaluate")
    e.add_argument("--detector", required=True, help="YOLO detect (or pose) weights for the boxes")
    e.add_argument("--weights", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--extra-gt", nargs="*", default=[], help="more test sets, as name=folder")
    e.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.out, args.epochs)
    else:
        evaluate(args.detector, args.weights, args.gt, args.name,
                 [tuple(x.split("=", 1)) for x in args.extra_gt], args.out)


if __name__ == "__main__":
    main()
