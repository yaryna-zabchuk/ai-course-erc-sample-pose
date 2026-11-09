"""Keypoint R-CNN (torchvision, two-stage) on the same two-point labels, for comparison with YOLO.

    python -m erc_pose.rcnn train --data data/processed_v2/pose --out runs/rcnn_v2
    python -m erc_pose.rcnn evaluate --weights runs/rcnn_v2/last.pt --gt data/processed_v2/pose --name rcnn_v2

The model has no order-free loss, so the ends are ordered left to right (angle in (-90, 90]).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from erc_pose.benchmark import list_images
from erc_pose.data.build_dataset import order_left_to_right
from erc_pose.evaluate import read_ground_truth, score_predictions
from erc_pose.orientation import angle_from_keypoints
from erc_pose.predict import Probe


def build_model(pretrained=True):
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

    # smaller anchors than the COCO default: many probes are under 30 px
    anchors = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.25, 1.0, 4.0),) * 5)
    model = keypointrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, rpn_anchor_generator=anchors,
                                      min_size=600, max_size=960, box_score_thresh=0.05)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, 2)
    return model


class PoseFolder(torch.utils.data.Dataset):
    """YOLO pose labels as torchvision detection targets."""

    def __init__(self, root, split, train=False):
        self.images = list_images(Path(root) / "images" / split)
        self.labels = Path(root) / "labels" / split
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        boxes, kpts = read_ground_truth(self.labels / f"{self.images[i].stem}.txt", width, height)
        keep = ~np.isnan(kpts).any((1, 2))
        boxes, kpts = boxes[keep], kpts[keep]
        if self.train:
            if np.random.rand() < 0.5:
                image = image[:, ::-1]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                kpts[..., 0] = width - kpts[..., 0]
            if np.random.rand() < 0.5:
                image = image[::-1]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                kpts[..., 1] = height - kpts[..., 1]
            image = np.clip(image * np.random.uniform(0.6, 1.4) + np.random.uniform(-25, 25), 0, 255)
        kpts = np.stack([order_left_to_right(k) for k in kpts]) if len(kpts) else kpts
        kpts[..., 2] = 1
        image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(boxes), dtype=torch.int64),
                  "keypoints": torch.tensor(kpts, dtype=torch.float32).reshape(-1, 2, 3)}
        return image, target


def train(data, out, epochs=24, batch=4, lr=0.01, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    loader = torch.utils.data.DataLoader(PoseFolder(data, "train", train=True), batch_size=batch, shuffle=True,
                                         num_workers=4, collate_fn=lambda b: tuple(zip(*b)))
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    steps = epochs * len(loader)
    warmup = min(500, len(loader))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    scaler = torch.amp.GradScaler()
    log = open(out / "train.log", "a")
    for epoch in range(epochs):
        model.train()
        total, start = 0.0, time.time()
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += loss.item()
        line = f"epoch {epoch + 1}/{epochs} loss {total / len(loader):.4f} {time.time() - start:.0f}s"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        torch.save(model.state_dict(), out / "last.pt")
    return out / "last.pt"


def load(weights, device):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image, device, conf=0.25):
    """Probes in a BGR image."""
    tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
    out = model([tensor.to(device)])[0]
    keep = out["scores"] >= conf
    probes = []
    for box, score, k in zip(out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy(),
                             out["keypoints"][keep].cpu().numpy()):
        ends = k[:, :2]
        probes.append(Probe(box, float(score), ends, float(angle_from_keypoints(ends[0], ends[1])), False))
    return probes


def evaluate(weights, gt, name, extra=(), out_dir="results", device="cuda"):
    model = load(weights, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    for set_name, folder in [(name, gt), *extra]:
        def run(path):
            image = cv2.imread(str(path))
            return predict(model, image, device), image.shape[:2]
        result = {"name": set_name, "task": "keypoint-rcnn", "weights": str(weights), "split": "test",
                  "params_m": params_m, **score_predictions(run, folder, "test")}
        if set_name == name:
            frames = [cv2.imread(str(p)) for p in list_images(Path(folder) / "images" / "test")[:100]]
            for dev, count in (("cuda", 100), ("cpu", 20)):
                m = load(weights, dev)
                for frame in frames[:3]:
                    predict(m, frame, dev)
                start = time.perf_counter()
                for frame in frames[:count]:
                    predict(m, frame, dev)
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
    t.add_argument("--epochs", type=int, default=24)
    t.add_argument("--batch", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("--weights", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--extra-gt", nargs="*", default=[], help="more test sets, as name=folder")
    e.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.out, args.epochs, args.batch)
    else:
        evaluate(args.weights, args.gt, args.name, [tuple(x.split("=", 1)) for x in args.extra_gt], args.out)


if __name__ == "__main__":
    main()
