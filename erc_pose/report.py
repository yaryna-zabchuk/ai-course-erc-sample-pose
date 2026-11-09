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


PART_COLUMNS = [
    ("params_m", "params, M", 1),
    ("recall", "recall", 2),
    ("precision", "precision", 2),
    ("map50", "mAP50", 2),
    ("angle_axis_mae_deg", "axis MAE, °", 1),
    ("angle_axis_median_deg", "median, °", 1),
    ("angle_axis_within_5deg", "≤5°", 2),
    ("testold.recall", "old: recall", 2),
    ("testold.angle_axis_mae_deg", "old: axis, °", 1),
    ("testnew.recall", "close-up: recall", 2),
    ("testnew.angle_axis_mae_deg", "close-up: axis, °", 1),
    ("testnew.angle_axis_within_5deg", "close-up: ≤5°", 2),
    ("fps_gpu", "FPS GPU", 0),
    ("fps_cpu", "FPS CPU", 0),
]


def parts_table(folder):
    """One row per model; <name>_testold / <name>_testnew results go into the same row."""
    rows = {}
    for path in sorted(folder.glob("*.json")):
        result = json.loads(path.read_text())
        if "name" not in result:  # e.g. tuned parameters
            continue
        name, _, part = result["name"].rpartition("_")
        if part in ("testold", "testnew"):
            rows.setdefault(name, {"name": name}).update({f"{part}.{k}": v for k, v in result.items()})
        else:
            rows.setdefault(result["name"], {}).update(result)
    columns = [("name", "model", None)] + PART_COLUMNS
    lines = ["| " + " | ".join(title for _, title, _ in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows.values():
        if "recall" in row:
            lines.append("| " + " | ".join(_cell(row.get(key), digits) for key, _, digits in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", default="results")
    parser.add_argument("--parts", action="store_true",
                        help="merge <name>_testold / <name>_testnew results into the row of <name>")
    args = parser.parse_args()
    folder = Path(args.results)
    if args.parts:
        text = parts_table(folder)
        (folder / "summary.md").write_text(text + "\n")
        print(text)
        return
    results = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    text = table(results)
    (folder / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
