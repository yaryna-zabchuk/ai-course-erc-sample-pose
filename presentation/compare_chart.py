"""Axis error vs speed for every model of the comparison.

    .venv/bin/python presentation/compare_chart.py presentation/fonts presentation
(uses Montserrat from the fonts folder when it is there, otherwise Inter)
reads results/v2/<model>_v2.json (axis error, recall) and results/v2/fps_check.json (end-to-end FPS)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONTS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
for f in FONTS.glob("*.ttf") if FONTS.exists() else []:
    font_manager.fontManager.addfont(str(f))
names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(n for n in ("Montserrat", "Inter", "DejaVu Sans") if n in names)

TEAL = "#00766B"
TEXT = "#0F3D38"
MUTED = "#4C6B66"
GREY = "#9AA9A6"
GRID = "#E3ECEA"

LABELS = {  # model name: (label on the chart, main model?)
    "pose": ("YOLOv8n-pose", True),
    "obb": ("YOLOv8n-OBB", True),
    "pose11": ("YOLO11n-pose", False),
    "pose11s": ("YOLO11s-pose", False),
    "obb11": ("YOLO11n-OBB", False),
    "seg": ("YOLOv8n-seg", False),
    "seg11": ("YOLO11n-seg", False),
    "rcnn": ("Keypoint R-CNN", False),
    "unet": ("U-Net", False),
    "twostage": ("YOLO + ResNet18", False),
    "classic": ("класичний CV", False),
    "lines": ("LSD-лінії", False),
}
OFFSETS = {  # label position in points from the dot; the YOLO cluster gets leader lines
    "pose": (40, -42), "obb": (-125, -42), "obb11": (48, -6), "pose11": (40, 22), "twostage": (-150, 10),
    "seg": (-125, -6), "seg11": (30, 14), "pose11s": (-115, 22),
}

results = Path("results/v2")
fps = json.loads((results / "fps_check.json").read_text())["fps"]
points = []
for name, (label, main) in LABELS.items():
    path = results / f"{name}_v2.json"
    if name not in fps or not path.exists():
        continue
    r = json.loads(path.read_text())
    points.append((name, label, main, fps[name], r["angle_axis_mae_deg"], r["recall"]))

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
for name, label, main, speed, error, recall in points:
    ax.scatter(speed, error, s=110 if main else 70, color=TEAL if main else GREY, zorder=3,
               edgecolor="white", linewidth=2)
    dx, dy = OFFSETS.get(name, (9, 5))
    ax.annotate(label, (speed, error), xytext=(dx, dy), textcoords="offset points", va="center",
                fontsize=12 if main else 10.5, fontweight="bold" if main else "regular",
                color=TEXT if main else MUTED,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=2, shrinkB=5) if name in OFFSETS else None)

ax.axhline(5, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
ax.text(5.3, 5.08, "ціль з пропоузалу: ≤ 5°", color=MUTED, fontsize=10)
ax.axvspan(25, 2000, color=TEAL, alpha=0.05, zorder=0)
ax.text(30, 0.25, "реальний час (≥ 25 FPS)", color=TEAL, fontsize=10)

ax.set_xscale("log")
ax.set_xlim(5, 2000)
ticks = [5, 10, 25, 50, 100, 200, 500, 1000]
ax.set_xticks(ticks, [str(t) for t in ticks])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(0, 5.6)
ax.set_xlabel("Швидкість, кадрів/с на ноутбуці (RTX 4050; класичні методи на CPU)", color=TEXT, fontsize=12)
ax.set_ylabel("Похибка кута осі, ° (менше — краще)", color=TEXT, fontsize=12)
ax.grid(True, color=GRID, lw=1, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=10)
ax.set_title("Точність кута проти швидкості: тест 156 кадрів, 194 проби", loc="left", color=TEAL,
             fontsize=15, fontweight="bold", pad=30)
ax.text(0, 1.025, "Нейромережі знаходять 86–93% проб; класичні методи — лише 46–53%, тому їхня точність кута "
        "рахується на легших пробах.", transform=ax.transAxes, color=MUTED, fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "compare_chart.png", dpi=150)
print(OUT / "compare_chart.png")
