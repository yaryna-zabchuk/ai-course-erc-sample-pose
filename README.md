# ERC probe detection and pose estimation

Detects the green cylindrical probes of the ERC Probing Task and estimates their
orientation in the image plane. Three YOLOv8 variants are trained on the same
split and compared: 2D boxes, two keypoints per probe (pose), and oriented
bounding boxes (OBB).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pipeline

The labelled data lives in the private Hugging Face dataset
`erc-samples/erc-sample-pose` (`hf auth login` first):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('erc-samples/erc-sample-pose', repo_type='dataset', allow_patterns=['semi_supervised_dataset/**'], local_dir='data/raw/erc-sample-pose')"
python -m erc_pose.data.build_dataset --yolo-pose data/raw/erc-sample-pose/semi_supervised_dataset
```

Each probe is labelled as its axis: a line between two keypoints (the box in the
export is just that line's extent). YOLO models need a box with an area to
train, so the build wraps each line in a thin box centred on it, with a width
of `--width-ratio` (0.18) x length. The width is only a training target: the
box's long axis is the labelled line, the model's output is turned back into a
line, and the angle errors are measured against the labelled lines. Box mAP
does depend on the assumed width, so treat it as a detection score, not an
angle score. Frames named `<camera>_<YYYYMMDD-HHMMSS-mmm>`
are grouped into sessions per camera, split where frames are 60 s or more apart.

Full pipeline from scratch:

```bash
# 1. frames from rover recordings (near-duplicates are skipped)
python -m erc_pose.data.extract_frames recordings/*.mp4 --out data/raw/frames

# 2. annotate in CVAT, export as "Ultralytics YOLO Pose 1.0" (with images), unzip, build the datasets
python -m erc_pose.data.build_dataset --yolo-pose data/raw/cvat_export
#    or from a COCO (Keypoints) JSON
python -m erc_pose.data.build_dataset --coco data/raw/annotations.json --images data/raw/frames

# 3. train
python -m erc_pose.train configs/baseline.yaml   # YOLOv8n, no augmentation
python -m erc_pose.train configs/detect.yaml
python -m erc_pose.train configs/pose.yaml
python -m erc_pose.train configs/obb.yaml

# 4. evaluate on the test split and build the comparison table
python -m erc_pose.evaluate --weights runs/pose/weights/best.pt --data data/processed/pose/data.yaml --name pose
python -m erc_pose.report results

# 5. run on the rover (obb_semi comes from the semi-supervised step below; runs/obb works too)
python -m erc_pose.infer --weights runs/obb_semi/weights/best.pt --source 0
```

No data yet? `python -m erc_pose.data.synthetic --out data/synthetic` writes
synthetic probes with the same annotation format, which is enough to run every
step above.

## Annotation conventions

- One class, `probe`, with a tight axis-aligned box.
- Two keypoints at the centres of the cylinder bases: `top` first, `bottom` second
  (in CVAT: a skeleton label whose points are created in that order).
- Either export works: Ultralytics YOLO Pose (or YOLO Detection for boxes only)
  or COCO (Keypoints). The export's own train/val subsets are ignored and the
  data is re-split.
- Frames are named `<recording>_<frame number>.jpg`. All frames of a recording
  go into the same split (70/15/15 by default), so near-duplicate frames cannot
  leak from training into the test set.

Annotations with boxes only produce the `detect` dataset. Once keypoints are
present, `pose` and `obb` are built as well; the oriented boxes are derived from
the keypoints and the box, so they need no extra labelling.

## Metrics

| Metric | Source |
|---|---|
| mAP50, mAP50-95 (boxes), pose mAP (OKS) | Ultralytics validation |
| keypoint MAE, px | `erc_pose/metrics.py` |
| angle MAE, ° and share within 5° | `erc_pose/metrics.py` |
| FPS on GPU and CPU | `erc_pose/benchmark.py` |

The orientation angle is `atan2(y2 - y1, x2 - x1)` from the top to the bottom
keypoint. Errors are circular, so 179° against -179° is 2°. An oriented box
cannot tell the two ends apart, so pose and OBB are compared on the *axis*
error (modulo 180°); the full 360° error is reported for pose models only.

The pose loss and pose mAP use an OKS sigma per keypoint (`--kpt-sigma`, default
0.5, the Ultralytics default for two keypoints). Smaller sigmas were tried and
made keypoints worse: the loss saturates on small probes and stops giving a
gradient. Pose mAP is comparable only between runs with the same sigma; the pixel
and angle errors are the sigma-independent measures.

## Label quality

The probe has a thick body and a thin tip, and `top` is the thick end. In the
first export about a quarter of the labels had the two points the other way
round, which made the keypoints collapse towards the middle of the probe.
`python -m erc_pose.data.check_direction <export> --out <fixed export>` measures
the green width near each end, swaps the labels that are clearly reversed and
writes `direction_report.csv`; short or unclear probes are listed as `unsure`
and should be checked by hand in CVAT.

## Semi-supervised training

`data/` in the Hugging Face dataset holds ~600 unlabelled frames. A trained OBB
model and pose model label them; a frame is kept only when every detection is
confident and the pose direction agrees with the OBB axis, and frames recorded
within 60 s of a val/test frame are skipped. The pseudo-labels then go into the
train split only:

```bash
python -m erc_pose.data.pseudo_label --weights runs/pose/weights/best.pt --obb-weights runs/obb/weights/best.pt \
    --images data/raw/erc-sample-pose/data --dataset data/processed/pose --out data/pseudo
python -m erc_pose.data.build_dataset --yolo-pose data/raw/semi_supervised_fixed --extra-train data/pseudo
```

## Direction classifier (experiment)

`erc_pose/direction.py` cuts each probe out along its axis and trains a tiny
YOLOv8n-cls to say which end is the top. It can replace the pose model as the
direction source (`--direction-weights runs/direction/weights/best.pt`), but on
the test split it chose the right end less often than the pose model (63-79% vs
82-84%), so the pose model stays the default.

```bash
python -m erc_pose.direction build --dataset data/processed/pose --out data/direction
python -m erc_pose.train configs/direction.yaml
```

## Cross-validation

The test split holds only ~20 probes, so single-split numbers move by several
points between runs. `erc_pose.crossval` makes every recording session the test
split once (5 folds) and reports mean ± std:

```bash
python -m erc_pose.crossval --export data/raw/semi_supervised_fixed --configs pose obb --fuse --name base
python -m erc_pose.crossval --export data/raw/semi_supervised_fixed --configs pose obb --fuse --name semi --extra-train data/pseudo
```

## Best setup so far

The team only needs the probe's axis, not which end is the top, so the
deliverable is a single OBB model trained with pseudo-labels. Its angle is the
axis in (-90°, 90°].

5-fold cross-validation over recording sessions (every human-labelled probe is
in a test fold once), mean ± std over folds:

| setup | axis error | axis ≤5° | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline: pose model, keypoints → axis | 17.8° ± 5.9 | 24% | 0.87 | 0.86 | 0.51 |
| OBB, human labels | 2.9° ± 0.2 | 85% | 0.91 | 0.94 | 0.61 |
| **OBB with pseudo-labels** | **2.3° ± 0.2** | **90% ± 4** | **0.93** | **0.95** | **0.64** |

Speed at 960 px, one frame at a time: 178 FPS on an RTX 4050 laptop GPU,
19 FPS on a 20-thread CPU.

```bash
python -m erc_pose.infer --weights runs/obb_semi/weights/best.pt --source 0
```

`scripts/run_all.sh` runs the whole pipeline: label fix, datasets, the four
models, pseudo-labels, the semi-supervised OBB and the evaluation.

### Keypoints with an order-free loss

Only the axis matters, so the two keypoints can be treated as interchangeable:
`configs/pose_sym.yaml` compares each prediction with both orderings of its
label and keeps the smaller loss (`erc_pose/symmetric.py`). 5-fold CV:

| pose model | axis error | axis ≤5° | recall |
|---|---|---|---|
| default loss, corrected labels | 17.8° ± 5.9 | 24% | 0.87 |
| left end always first (`--kpt-order left-to-right`) | 31.0° ± 3.7 | 12% | 0.91 |
| order-free loss, corrected labels | 6.3° ± 3.2 | 65% | 0.94 |
| **order-free loss, original labels** | **5.3° ± 2.2** | **69%** | **0.94** |
| order-free loss, original labels + pseudo-labels | 5.2° ± 3.2 | 72% | 0.91 |

Pseudo-labels helped on 4 of 5 folds but made one fold much worse, so the
mean barely moves. The final model with pseudo-labels (`runs/pose_sym_semi`)
gets 3.3° (median 2.2°) and 79% within 5° on the test split, against 1.7° and
94% for the OBB.

With the order-free loss the label direction no longer matters, so the
original export works without `check_direction`. A fixed left-to-right order
is worse because near-vertical probes flip between the two orders. OBB is still
more accurate (2.3-2.9°).

### Drawn boxes

The team also drew a box around each probe (CVAT export in YOLO Detection
format). `build_dataset --boxes <export>` attaches each drawn box to the axis
line inside it; the detect and pose models then train on the drawn boxes and the
OBB gets each probe's real width. 5-fold CV, same folds as above:

| model | estimated boxes | drawn boxes |
|---|---|---|
| OBB: axis error / ≤5° / recall | 2.9° ± 0.2 / 85% / 0.91 | 3.3° ± 0.4 / 83% / 0.93 |
| pose, order-free: axis error / ≤5° / recall | 6.3° ± 3.2 / 65% / 0.94 | 4.7° ± 1.1 / 66% / 0.94 |

Drawn boxes help the pose model (lower and much steadier axis error) and
slightly raise recall, but not the OBB axis: the drawn widths vary a lot
(0.13-0.51 of the length), and a loose box around a short probe makes the
oriented box nearly square, so its long side is less clear.

```bash
python -m erc_pose.data.build_dataset --yolo-pose data/raw/semi_supervised_fixed \
    --boxes data/raw/box_labels_task2660063 --out data/processed_boxes
```

### If the direction is needed later

A pose model can add the top/bottom direction to the OBB axis
(`--direction-weights runs/pose/weights/best.pt`). In cross-validation that got
the right end for 89% of probes and the full angle within 5° for 80%, at half
the speed (87 / 10 FPS). The pose model for this must be trained on human labels
only; trained on pseudo-labels it copies the teacher's direction mistakes (84%
right). Flip test-time augmentation (`--direction-tta`) made the direction
slightly worse (86.5% ± 6 instead of 88.9% ± 3). About a quarter of the probes
that `check_direction` marks `unsure` are probably still labelled the wrong way
round, which limits the direction accuracy.
