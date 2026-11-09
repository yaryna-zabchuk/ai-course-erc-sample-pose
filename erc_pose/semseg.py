"""Semantic segmentation: a U-Net marks probe pixels, each blob of pixels is one probe.

    python -m erc_pose.semseg train --data data/processed_v2_est/obb --out runs/unet_v2
    python -m erc_pose.semseg evaluate --weights runs/unet_v2/last.pt --gt data/processed_v2/pose --name unet_v2

No boxes and no instances: the masks are the OBB polygons of the labels, the axis of a
blob is the main direction of its pixels (PCA).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from erc_pose.benchmark import list_images
from erc_pose.evaluate import score_predictions
from erc_pose.lines import box_around
from erc_pose.orientation import angle_from_keypoints
from erc_pose.predict import Probe

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class UNet(nn.Module):
    """U-Net with a ResNet18 encoder; skips at strides 2-16 keep thin probes."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18

        r = resnet18(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)                     # /2, 64
        self.enc = nn.ModuleList([nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4])  # /4../32
        self.dec = nn.ModuleList([self.block(512 + 256, 256), self.block(256 + 128, 128),
                                  self.block(128 + 64, 64), self.block(64 + 64, 64)])
        self.head = nn.Sequential(self.block(64 + 3, 32), nn.Conv2d(32, 1, 1))

    @staticmethod
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                             nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                             nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % 32, 0, -h % 32))
        skips = [x, self.stem(x)]
        for layer in self.enc:
            skips.append(layer(skips[-1]))
        y = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            y = block(torch.cat([F.interpolate(y, size=skip.shape[-2:], mode="bilinear"), skip], 1))
        y = self.head(torch.cat([F.interpolate(y, size=x.shape[-2:], mode="bilinear"), skips.pop()], 1))
        return y[..., :h, :w]


def read_mask(label_path, width, height):
    """Probe mask from YOLO OBB / polygon labels."""
    mask = np.zeros((height, width), np.uint8)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = np.array(line.split()[1:], dtype=float)
            if len(values) >= 6:
                cv2.fillPoly(mask, [np.round(values.reshape(-1, 2) * [width, height]).astype(np.int32)], 1)
    return mask


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class MaskFolder(torch.utils.data.Dataset):
    def __init__(self, root, split, train=False):
        self.images = list_images(Path(root) / "images" / split)
        self.labels = Path(root) / "labels" / split
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = read_mask(self.labels / f"{self.images[i].stem}.txt", w, h)
        if self.train:
            s = np.random.uniform(0.7, 1.4)  # zoom in or out, then crop / pad back to the frame size
            image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            ph, pw = max(h - image.shape[0], 0), max(w - image.shape[1], 0)
            image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            y0 = np.random.randint(0, image.shape[0] - h + 1)
            x0 = np.random.randint(0, image.shape[1] - w + 1)
            image, mask = image[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w]
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                image, mask = image[::-1], mask[::-1]
            image = np.clip(image * np.random.uniform(0.6, 1.4) + np.random.uniform(-25, 25), 0, 255)
        return to_tensor(image), torch.from_numpy(mask[None].astype(np.float32).copy())


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(5.0, device=logits.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def train(data, out, epochs=40, batch=4, lr=5e-4, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    loader = torch.utils.data.DataLoader(MaskFolder(data, "train", train=True), batch_size=batch, shuffle=True,
                                         num_workers=4, drop_last=True,
                                         worker_init_fn=lambda i: np.random.seed(torch.initial_seed() % 2**32))
    val = torch.utils.data.DataLoader(MaskFolder(data, "val"), batch_size=batch, num_workers=2)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs * len(loader), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    log = open(out / "train.log", "a")
    for epoch in range(epochs):
        model.train()
        total, start = 0.0, time.time()
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type="cuda"):
                loss = loss_fn(model(images).float(), masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += loss.item()
        model.eval()
        inter = union = 0.0
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            for images, masks in val:
                pred = (model(images.to(device)) > 0).float()
                masks = masks.to(device)
                inter += (pred * masks).sum().item()
                union += ((pred + masks) > 0).sum().item()
        line = (f"epoch {epoch + 1}/{epochs} loss {total / len(loader):.4f} val IoU {inter / max(union, 1):.3f} "
                f"{time.time() - start:.0f}s")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        torch.save(model.state_dict(), out / "last.pt")
    return out / "last.pt"


def load(weights, device):
    model = UNet(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def blobs_to_probes(prob, threshold=0.5, min_area=15):
    """One probe per connected blob of probe pixels; axis from PCA of the pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob > threshold).astype(np.uint8), connectivity=8)
    probes = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(labels == c)
        pts = np.c_[xs, ys].astype(float)
        centre = pts.mean(0)
        _, vectors = np.linalg.eigh(np.cov((pts - centre).T))
        d = vectors[:, -1]  # largest eigenvalue: the long axis
        proj = (pts - centre) @ d
        ends = np.stack([centre + proj.min() * d, centre + proj.max() * d])
        conf = float(prob[ys, xs].mean())
        probes.append(Probe(box_around(ends), conf, ends, float(angle_from_keypoints(ends[0], ends[1])), False))
    return probes


@torch.no_grad()
def predict(model, image, device):
    """Probes in a BGR image."""
    x = to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[None].to(device)
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
    return blobs_to_probes(prob)


def evaluate(weights, gt, name, extra=(), out_dir="results", device="cuda"):
    model = load(weights, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    for set_name, folder in [(name, gt), *extra]:
        def run(path):
            image = cv2.imread(str(path))
            return predict(model, image, device), image.shape[:2]
        result = {"name": set_name, "task": "unet", "weights": str(weights), "split": "test", "params_m": params_m,
                  **score_predictions(run, folder, "test")}
        if set_name == name:
            frames = [cv2.imread(str(p)) for p in list_images(Path(folder) / "images" / "test")[:100]]
            for dev, count in (("cuda", 100), ("cpu", 30)):
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
    t.add_argument("--data", required=True, help="OBB or seg dataset folder (polygon labels)")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=40)
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
