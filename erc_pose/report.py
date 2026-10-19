"""Combine the evaluation JSON files into one comparison table.

    python -m erc_pose.report results
"""

import argparse
import json
import math
from pathlib import Path

COLUMNS = [
    ("name", "model", None),
    ("map50", "mAP50", 3),
    ("map50_95", "mAP50-95", 3),
    ("pose_map50_95", "pose mAP50-95", 3),
    ("kpt_mae_px", "kpt MAE, px", 2),
    ("angle_mae_deg", "angle MAE, °", 2),
    ("angle_within_5deg", "angle ≤5°", 3),
    ("directed_share", "directed", 2),
    ("direction_acc", "top/bottom ok", 3),
    ("angle_axis_mae_deg", "axis MAE, °", 2),
    ("angle_axis_within_5deg", "axis ≤5°", 3),
    ("fps_gpu", "FPS GPU", 1),
    ("fps_cpu", "FPS CPU", 1),
]


def _cell(value, digits):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return str(value) if digits is None else f"{value:.{digits}f}"


def table(results):
    lines = ["| " + " | ".join(title for _, title, _ in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for row in results:
        lines.append("| " + " | ".join(_cell(row.get(key), digits) for key, _, digits in COLUMNS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", default="results")
    args = parser.parse_args()
    folder = Path(args.results)
    results = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    text = table(results)
    (folder / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
