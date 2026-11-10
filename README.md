# ERC probe detection and orientation

Finds the green probes of the ERC Probing Task in rover camera frames and
estimates the angle of each probe's axis in the image. The model is
YOLOv8n-pose with two keypoints per probe (the two ends); the angle is
`atan2(y2 - y1, x2 - x1)`, taken as an axis, so it lies in (-90°, 90°].

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Models

All four are YOLOv8n at 960 px, trained on the same 735 frames and tested on the
same split (110 frames, 153 probes). Weights are in `models/`.

| model | what it does | config | weights | test results |
|---|---|---|---|---|
| baseline | box detection, no augmentation | `configs/baseline.yaml` | `models/baseline.pt` | mAP50 0.82, recall 0.81, precision 0.84 |
| box detector | box detection, with augmentation | `configs/detect.yaml` | `models/detect.pt` | mAP50 0.88, recall 0.90, precision 0.80 |
| **pose (main)** | box + two points per probe → axis angle | `configs/pose_sym.yaml` | `models/pose.pt` | **axis error 3.2° (median 2.3°), 82% within 5°**, mAP50 0.92, recall 0.91, precision 0.83 |
| OBB (alternative) | rotated box per probe → axis angle | `configs/obb.yaml` | `models/obb.pt` | **axis error 2.1° (median 1.5°), 91% within 5°**, recall 0.85, precision 0.90 |

The pose and OBB models give the angle; the baseline and the box detector only
find the probes. All run in real time: 160-185 FPS on an RTX 4050 laptop GPU,
15-20 FPS on CPU. `configs/experiments/` holds configs from earlier experiments
that are not used for these models.

### OBB

The proposal planned YOLOv8-OBB as the alternative to keypoints. An oriented
bounding box is a rectangle rotated to lie along the probe, and the probe's axis
is its long side. Nobody draws OBBs: each label line is turned into a thin
rotated rectangle (width 0.18 × length) when the dataset is built, and the
predicted box is turned back into a line for scoring, so both models are judged
against the same labelled lines.

On the test split OBB gets the angle more precisely than the pose model (2.1° vs
3.2°, 91% vs 82% within 5°) but finds fewer probes (recall 0.85 vs 0.91). Its
mAP is left out of the table because its label boxes differ from those of the
other models. The drawn boxes do not help OBB: their widths vary a lot (0.13-0.51
of the length) and a loose box around a short probe makes the rotated rectangle
almost square, so OBB is trained on the generated boxes only. Pose stays the
main model, as planned in the proposal; OBB is the more accurate option when
only the angle matters.

## Run it

Every model of the comparison runs through the same command; `python -m erc_pose.models`
lists them (`pose`, `obb`, `pose11`, `pose11s`, `obb11`, `seg`, `seg11`, `detect`,
`baseline`, `rcnn`, `unet`, `twostage`, `classic`, `lines`) and whether their weights are there.

```bash
# folder of photos or a video, annotated frames saved to out/
python -m erc_pose.infer --model pose --source path/to/photos --save out

# webcam (Esc or q to quit)
python -m erc_pose.infer --model pose --source 0 --show

# rover cameras (Linux); the front-bottom camera is mounted upside down
python -m erc_pose.infer --model pose --source /dev/arducam-mast --show
python -m erc_pose.infer --model obb --source /dev/arducam-front-bottom --rotate 180 --show

# any Ultralytics weights work too
python -m erc_pose.infer --weights models/pose.pt --source 0 --show
```

Every frame prints one JSON line, e.g.
`{"frame": 0, "model": "pose", "probes": [{"bbox": [...], "conf": 0.85, "kpts": [...], "axis_deg": 12.3, ...}]}`;
`axis_deg` is the axis angle in (-90°, 90°]. Cameras are captured at 960×600, the
size of the training frames. `--save-raw` keeps every 10th clean frame for
labelling, `--conf` sets the detection threshold (0.25).

### Field test

```bash
scripts/field.sh mast pose obb unet        # n / p switch model, q quits
scripts/field.sh front pose                # front-bottom camera, rotated 180°
python -m erc_pose.models --check path/to/frames   # load every model, measure FPS
```

`field.sh` shows the chosen models live on a rover camera (`mast`, `front`, `rear`
or a `/dev/videoN` path) and keeps everything in `field/<time>_<camera>/`: one
JSON line per frame, annotated frames with a probe, and every 10th clean frame.

## Data

- 141 frames labelled by hand: the Hugging Face dataset `erc-samples/erc-sample-pose`
  (`semi_supervised_dataset/`, YOLO Pose export).
- 594 more frames (the `data/` folder of the same dataset), pre-labelled by the
  model and corrected in CVAT (task 2686785, YOLO Pose export).
- Boxes drawn around the probes of the first 141 frames (task 2660063,
  YOLO Detection export).

Together: 735 frames, 781 probes, three cameras (mast, front-bottom, arm).

Labels: each probe is a `sample_pose` skeleton with two points, one at each end
of the probe. The order of the points does not matter. A `sample_box` rectangle
is optional; where it is missing, a box is built around the line
(width 0.18 × length).

```bash
hf auth login
python -c "from huggingface_hub import snapshot_download; snapshot_download('erc-samples/erc-sample-pose', repo_type='dataset', local_dir='data/raw/erc-sample-pose')"

python -m erc_pose.data.build_dataset \
    --yolo-pose data/raw/erc-sample-pose/semi_supervised_dataset path/to/task_2686785_export \
    --images data/raw/erc-sample-pose/data \
    --boxes path/to/task_2660063_export \
    --min-line-px 3 --out data/processed_all
```

The split is 70/15/15 by recording session (camera + frames less than 60 s
apart), never by frame, since neighbouring frames are near-duplicates:
515 / 110 / 110 frames.

## Train

```bash
python -m erc_pose.train configs/baseline.yaml --data data/processed_all/detect/data.yaml --set name=baseline
python -m erc_pose.train configs/detect.yaml --data data/processed_all/detect/data.yaml --set name=detect
python -m erc_pose.train configs/pose_sym.yaml --data data/processed_all/pose/data.yaml --set name=pose

# OBB uses the generated boxes only: same build without --boxes
python -m erc_pose.data.build_dataset \
    --yolo-pose data/raw/erc-sample-pose/semi_supervised_dataset path/to/task_2686785_export \
    --images data/raw/erc-sample-pose/data --min-line-px 3 --out data/processed_all_est
python -m erc_pose.train configs/obb.yaml --data data/processed_all_est/obb/data.yaml --set name=obb
```

`scripts/run_all.sh` builds the dataset, trains the baseline, box detector and
pose model and evaluates them.

`configs/pose_sym.yaml` uses a keypoint loss that ignores which end of the
probe comes first (`erc_pose/symmetric.py`), trains all 100 epochs and the
weights to use are `weights/last.pt` (Ultralytics' `best.pt` can be an early
epoch whose points have not reached the probe ends yet). Augmentation is done on
the fly: brightness, colour, zoom/crop, mosaic, flips, noise, blur. No rotation,
it inflates the boxes of thin objects.

## Evaluate and benchmark

```bash
python -m erc_pose.evaluate --weights models/pose.pt --data data/processed_all/pose/data.yaml --name pose
python -m erc_pose.evaluate --weights models/obb.pt --data data/processed_all_est/obb/data.yaml --name obb
python -m erc_pose.benchmark --weights models/pose.pt --images data/processed_all/pose/images/test
python -m erc_pose.report results
```

`evaluate` writes `results/<name>.json`: mAP50 / mAP50-95, recall, precision,
axis error in degrees (mean, median, share within 5°) and FPS on GPU and CPU.
`erc_pose.crossval` runs 5-fold cross-validation over recording sessions.

## Results

The main numbers are in the table under [Models](#models). For box models
`evaluate` needs the box labels as ground truth: add `--gt data/processed_all/detect`.
Speed is measured one frame at a time at 960 px (GPU: RTX 4050 laptop, CPU: 20 threads).

How the pose model got there (5-fold cross-validation on the first 141 frames):

| | axis error | within 5° |
|---|---|---|
| baseline: YOLOv8n-pose, standard loss | 17.8° ± 5.9 | 24% |
| keypoint loss without point order | 5.3° ± 2.2 | 69% |
| + drawn boxes | 4.7° ± 1.1 | 66% |
| + pseudo-labels, last epoch | 3.9° ± 1.4 | 78% |

On the test frames that neither model saw in training,
the model trained on all 735 frames beats the earlier one: 2.9° vs 4.0° axis
error, recall 0.88 vs 0.80.

Known weak spots: very close probes (the thin tip gets cut off), probes under
~15 px, and occasional false hits on lens glare and screens.

## Model comparison with close-up frames

294 more frames (PNG, 960×600) from the front-bottom and arm cameras, with the
probe close to the camera: median probe length 146 px against about 40 px in the
first 735 frames. 273 have one probe, 21 have none. They are split by session on
their own (`data/raw/closeup_20261108/split.txt`, 201 / 47 / 46 frames, both
cameras in every split) and added to the split of the first 735 frames, which
stays as it was:

```bash
python -m erc_pose.data.build_dataset --yolo-pose path/to/closeup_export --min-line-px 3 \
    --out data/processed_closeup
python scripts/add_closeup.py data/processed_all data/processed_closeup \
    data/raw/closeup_20261108/split.txt data/processed_v2
python scripts/add_closeup.py data/processed_all_est data/processed_closeup \
    data/raw/closeup_20261108/split.txt data/processed_v2_est
scripts/run_v2.sh        # trains and evaluates every model below
python -m erc_pose.report results/v2 --parts
```

Test: 156 frames, 194 probes; "old" is the 153 probes of the earlier test split,
"close-up" the 41 new ones. All models are trained on the same 716 frames.

| model | params | recall | precision | axis error (median) | ≤5° | close-up axis error | FPS GPU / CPU |
|---|---|---|---|---|---|---|---|
| classic CV: green threshold + rotated rectangle, no learning | – | 0.53 | 0.83 | 2.5° (1.9°) | 89% | 1.7° | – / 661 |
| classic CV: line segments (LSD) on green pixels, no learning | – | 0.46 | 0.80 | 4.0° (2.8°) | 77% | 4.4° | – / 21 |
| baseline: YOLOv8n detect, no augmentation | 3.1M | 0.89 | 0.86 | – | – | – | 173 / 16 |
| YOLOv8n detect | 3.1M | 0.91 | 0.86 | – | – | – | 172 / 16 |
| **YOLOv8n-pose (main)** | 3.1M | 0.88 | 0.85 | **2.2° (1.3°)** | 88% | 1.7° | 181 / 15 |
| YOLO11n-pose | 2.7M | 0.88 | 0.86 | 2.6° (2.0°) | 87% | 1.9° | 183 / 18 |
| YOLO11s-pose | 9.7M | 0.93 | 0.87 | 2.7° (1.6°) | 87% | 2.4° | 87 / 6 |
| **YOLOv8n-OBB** | 3.1M | 0.89 | 0.86 | **2.1° (1.4°)** | 91% | 1.5° | 173 / 16 |
| YOLO11n-OBB | 2.7M | 0.88 | 0.90 | 2.1° (1.5°) | 91% | 1.7° | 176 / 17 |
| YOLOv8n-seg, angle from the mask | 3.4M | 0.91 | 0.82 | 3.2° (2.2°) | 85% | 2.2° | 136 / 11 |
| YOLO11n-seg | 2.9M | 0.91 | 0.83 | 3.5° (2.4°) | 83% | 2.3° | 142 / 13 |
| Keypoint R-CNN (ResNet-50 FPN) | 59M | 0.91 | 0.60 | 3.8° (1.6°) | 88% | 6.7° | 11 / 1 |
| U-Net (ResNet18 encoder), probe pixels → blobs → PCA axis | 14.4M | 0.86 | 0.87 | 2.3° (1.4°) | 90% | 1.7° | 24 / 2 |
| two-stage: YOLOv8n detect + ResNet18 angle per crop | 14.3M | 0.91 | 0.86 | 2.5° (1.6°) | 87% | 2.4° | 117 / 14 |

For comparison, the models trained without the close-ups on the same test:
pose 3.2° (2.3°), 82% within 5°, close-up 3.3°; OBB 2.1° (1.5°), 92%, close-up 1.9°.

- The close-ups cut the pose model's error from 3.2° to 2.2° (1.7° on close
  probes). Recall on the old frames fell from 0.91 to 0.86; the missed probes
  are found, but in a few sunlit frames of one session the box stops before the
  thin tip, and IoU drops just under 0.5.
- OBB and pose are now about level on the angle; OBB is still slightly ahead.
- YOLO11 is not better than YOLOv8 here. The bigger YOLO11s finds the most
  probes (recall 0.93) but runs at half the speed.
- Segmentation finds probes well but the angle from the mask is less precise.
- Keypoint R-CNN is 19× bigger and 16× slower with more false hits; its angle
  has large outliers on close probes (heatmap resolution, and the left-to-right
  point order flips for near-vertical probes).
- The classic baseline gets the angle right when it finds a probe, but misses
  half of them (small, shaded or partly hidden probes) and has no notion of
  what is a probe beyond colour and shape.

Each row is one training run on one test split (~200 probes), so differences of
a few tenths of a degree are noise. Cross-validation over recording sessions on
all 1029 frames (3 of 5 folds, `scripts/run_cv_v2.sh`) settles pose vs OBB:

| | axis error | within 5° | recall | mAP50 |
|---|---|---|---|---|
| YOLOv8n-pose | 2.41° ± 0.19 | 91% | 0.945 ± 0.009 | 0.95 |
| YOLOv8n-OBB | 1.61° ± 0.02 | 95% | 0.934 ± 0.017 | 0.95 |

OBB gives the more precise axis; the gap is several times the spread over folds.

- The U-Net, which knows nothing about boxes, gets close to the YOLO models
  (2.3°, 90% within 5°) but runs at 24 FPS: it labels every pixel of the full frame.
  It can split a probe in two where the middle is in shadow.
- The two-stage model keeps the detector's recall (0.91) and adds an angle
  regressor that can be trained with any rotation; its angle (2.5°) does not beat
  pose or OBB, which see the whole probe in context.

U-Net and two-stage: `scripts/run_v2_extra.sh` trains and evaluates both.
`presentation/compare_chart.py` draws axis error against speed, and
`scripts/compare_grid.py` shows the same test frames through several models side by side.

Classic baseline: `python -m erc_pose.classic --gt data/processed_v2/pose --name classic_v2`
(tunes its thresholds on train + val). Keypoint R-CNN: `python -m erc_pose.rcnn train` /
`evaluate` (torchvision; points ordered left to right, since it has no
order-free loss).

## Pre-labelling new frames

```bash
python -m erc_pose.data.prelabel --weights models/pose.pt \
    --images path/to/new/frames --labelled data/raw/erc-sample-pose/semi_supervised_dataset \
    --out data/cvat_prelabels
```

Produces a zip that CVAT imports as "Ultralytics YOLO Pose 1.0".
