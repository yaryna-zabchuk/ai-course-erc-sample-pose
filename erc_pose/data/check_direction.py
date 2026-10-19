"""Find probes whose top/bottom keypoints were swapped by the annotator.

The probe has a thick cylindrical body and a thin pointed tip; keypoint 0
(`top`) belongs on the thick end. The green width of the probe near each end
tells them apart. Labels that are clearly the other way round are listed,
and with `--out` a copy of the YOLO Pose export is written with them swapped.
Short probes and unclear width profiles are left alone and listed as unsure.

    python -m erc_pose.data.check_direction data/raw/erc-sample-pose/semi_supervised_dataset \
        --out data/raw/semi_supervised_fixed
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from erc_pose.benchmark import IMAGE_SUFFIXES

GREEN_LOW, GREEN_HIGH = (35, 60, 50), (90, 255, 255)  # OpenCV HSV


def end_widths(green, p1, p2, samples=(0.15, 0.2, 0.25)):
    """Green width (px) across the probe near p1 and near p2."""
    height, width = green.shape
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    axis = (p2 - p1) / length
    normal = np.array([-axis[1], axis[0]])
    offsets = np.arange(-0.3 * length, 0.3 * length, 0.5)

    def width_at(t):
        points = np.round(p1 + axis * length * t + normal * offsets[:, None]).astype(int)
        inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        points = points[inside]
        return green[points[:, 1], points[:, 0]].sum() * 0.5

    return (float(np.mean([width_at(t) for t in samples])),
            float(np.mean([width_at(1 - t) for t in samples])))


def classify(widths, length, min_length=25.0, min_width=2.0, margin=1.4):
    """'ok', 'swapped' or 'unsure' for one probe."""
    top, bottom = widths
    if length < min_length or max(top, bottom) < min_width:
        return "unsure"
    if top >= margin * bottom:
        return "ok"
    if bottom >= margin * top:
        return "swapped"
    return "unsure"


def check(export_dir, out_dir=None, report=None):
    export_dir = Path(export_dir)
    images = {p.stem: p for p in export_dir.glob("images/**/*") if p.suffix.lower() in IMAGE_SUFFIXES}
    rows, fixed = [], {}
    for label in sorted(export_dir.glob("labels/**/*.txt")):
        image = cv2.imread(str(images[label.stem]))
        height, width = image.shape[:2]
        green = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), GREEN_LOW, GREEN_HIGH) > 0
        lines = []
        for index, line in enumerate(label.read_text().splitlines()):
            values = line.split()
            if len(values) != 11:
                lines.append(line)
                continue
            p1 = np.array([float(values[5]) * width, float(values[6]) * height])
            p2 = np.array([float(values[8]) * width, float(values[9]) * height])
            length = float(np.linalg.norm(p2 - p1))
            widths = end_widths(green, p1, p2) if length > 0 else (0.0, 0.0)
            verdict = classify(widths, length)
            rows.append({"image": label.stem, "probe": index, "length_px": round(length, 1),
                         "top_width_px": round(widths[0], 1), "bottom_width_px": round(widths[1], 1),
                         "verdict": verdict})
            if verdict == "swapped":
                values[5:8], values[8:11] = values[8:11], values[5:8]
            lines.append(" ".join(values))
        fixed[label] = "\n".join(lines) + "\n"

    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("ok", "swapped", "unsure")}
    print(", ".join(f"{k}: {v}" for k, v in counts.items()))
    if report:
        with open(report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if out_dir:
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(export_dir, out_dir, ignore=shutil.ignore_patterns("*.txt", ".DS_Store"))
        for label, text in fixed.items():
            target = out_dir / label.relative_to(export_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="YOLO Pose export folder (images/ and labels/)")
    parser.add_argument("--out", help="write a copy with the swapped labels corrected")
    parser.add_argument("--report", default="direction_report.csv", help="CSV with one row per probe")
    args = parser.parse_args()
    check(args.export, args.out, args.report)


if __name__ == "__main__":
    main()
