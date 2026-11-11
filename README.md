# ERC probe detection and orientation

Finds the green probes of the ERC Probing Task in rover camera frames and gives
the angle of each probe's axis in the image, in (-90°, 90°]. The axis has no
direction: the models do not say which end is the top.

**Quick start** (after [Setup](#setup)):

```bash
python -m erc_pose.models                       # which models there are and whether their weights are present
scripts/field.sh mast obb                       # the most precise model, live on the mast camera
TILE=1 scripts/field.sh mast obb pose           # two models side by side on the same frames
```

Contents: [Setup](#setup) · [Which model to use](#which-model-to-use) ·
[How each model works](#how-each-model-works) · [Run it](#run-it) · [Data](#data) ·
[Train](#train) · [Evaluate](#evaluate) · [Results](#results) ·
[Pre-labelling new frames](#pre-labelling-new-frames)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Weights live in `models/` (not in git; ask AngryBandera or take them from the
Hugging Face model repo once it is uploaded).

## Which model to use

Every model has a short name that the run commands take. Numbers: test split of
156 frames / 194 probes (old frames + close-ups); FPS end to end on an RTX 4050
laptop GPU and on the CPU (`results/v2/fps_check*.json`).

| name | model | axis error | within 5° | recall | FPS GPU / CPU | use it for |
|---|---|---|---|---|---|---|
| **`obb`** | YOLOv8n-OBB | **2.1°** | **91%** | 0.89 | 161 / 16 | **default**: the most precise axis (cross-validation 1.61° ± 0.02) |
| **`pose`** | YOLOv8n-pose | 2.2° | 88% | 0.88 | 176 / 15 | **main model of the proposal**; two end points |
| `obb11` | YOLO11n-OBB | 2.1° | 91% | 0.88 | 171 / 20 | as `obb`, a bit faster on CPU |
| `pose11` | YOLO11n-pose | 2.6° | 87% | 0.88 | 179 / 20 | as `pose`, a bit faster on CPU |
| `pose11s` | YOLO11s-pose | 2.7° | 87% | **0.93** | 82 / 7 | finds the most probes; half the speed |
| `twostage` | YOLOv8n detect + ResNet18 angle | 2.5° | 87% | 0.91 | 119 / 16 | comparison |
| `seg` / `seg11` | YOLOv8n-seg / YOLO11n-seg | 3.2° / 3.5° | 85% / 83% | 0.91 | 128 / 13, 136 / 14 | comparison |
| `unet` | U-Net, pixels → PCA | 2.3° | 90% | 0.86 | 24 / 1.5 | comparison; slow |
| `rcnn` | Keypoint R-CNN | 3.8° | 88% | 0.91 | 12 / 0.9 | comparison; slow, many false hits |
| `classic` | green threshold, no learning | 2.5° | 89% | 0.53 | 949 (CPU) | baseline without learning |
| `lines` | LSD line segments, no learning | 4.0° | 77% | 0.46 | 28 (CPU) | baseline without learning |
| `detect` | YOLOv8n box detector | – | – | 0.91 | 174 / 18 | boxes only, no angle |
| `baseline` | YOLOv8n box detector, no augmentation | – | – | 0.89 | 172 / 18 | boxes only, no angle |

In short:
- **On the rover: `obb`.** If you need the two end points, use `pose`.
- **On a CPU only:** `obb` and `pose` still run at 15-16 FPS; `obb11` and
  `pose11` at about 20 FPS.
- **Don't use `unet`, `rcnn` or `lines` for driving.** They are below 25 FPS
  on the GPU and are there for the comparison.

Weights per name (`erc_pose/models.py`, `MODELS`):

| name | weights | config | trained on |
|---|---|---|---|
| `pose`, `pose11`, `pose11s` | `models/pose_v2.pt`, `pose11_v2.pt`, `pose11s_v2.pt` | `configs/pose_sym.yaml`, `pose11.yaml`, `pose11s.yaml` | `data/processed_v2/pose` |
| `obb`, `obb11` | `models/obb_v2.pt`, `obb11_v2.pt` | `configs/obb.yaml`, `obb11.yaml` | `data/processed_v2_est/obb` |
| `seg`, `seg11` | `models/seg_v2.pt`, `seg11_v2.pt` | `configs/seg.yaml`, `seg11.yaml` | `data/processed_v2_est/seg` |
| `detect`, `baseline` | `models/detect_v2.pt`, `baseline_v2.pt` | `configs/detect.yaml`, `baseline.yaml` | `data/processed_v2/detect` |
| `rcnn` | `models/rcnn_v2.pt` | `erc_pose/rcnn.py` | `data/processed_v2/pose` |
| `unet` | `models/unet_v2.pt` | `erc_pose/semseg.py` | `data/processed_v2_est/obb` (masks) |
| `twostage` | `models/detect_v2.pt` + `models/angle_v2.pt` | `erc_pose/twostage.py` | `data/processed_v2/pose` |
| `classic`, `lines` | `models/classic_params.json`, `lines_params.json` | `erc_pose/classic.py`, `lines.py` | thresholds tuned on train + val |

`models/pose.pt`, `obb.pt`, `detect.pt` and `baseline.pt` are the earlier models,
trained without the close-up frames; run them with `--weights models/pose.pt`.

## How each model works

All models learn from the same labels: one line per probe, from the end of
the thick body to the tip. Each model's output is turned back into a line, and
every model is scored against the labelled lines.

**YOLO pose (`pose`, `pose11`, `pose11s`).**
- Predicts a box and two points per probe, the two ends; the axis is the line
  between them.
- Uses an order-free keypoint loss (`erc_pose/symmetric.py`). The labelled point
  order is not consistent, and the standard loss pulled both points to the
  probe's middle.
- Weights from the last epoch (`last.pt`). Ultralytics' `best.pt` can be an
  early epoch whose points have not reached the ends yet.

**YOLO OBB (`obb`, `obb11`).**
- Predicts a rotated box; its long side is the axis.
- The rotated boxes for training are built from the labelled line (width
  0.18 × length), so nobody draws them.
- Predicts the angle directly and has no points that can collapse. This makes
  it the most precise and stable model.

**YOLO segmentation (`seg`, `seg11`).**
- Predicts a pixel mask per probe; the axis is the smallest rotated rectangle
  around the mask.
- Trained on the same rotated rectangles as OBB.
- The mask is coarse (160×160 scaled up), so the angle is less precise.

**Two-stage (`twostage`).**
- A YOLOv8n detector finds upright boxes. Each box is cropped (+25% margin,
  128×128), and a ResNet18 predicts the angle as (cos 2θ, sin 2θ), which
  treats 0° and 180° as the same axis.
- The angle network is trained on randomly rotated crops.

**Keypoint R-CNN (`rcnn`).**
- torchvision's two-stage keypoint model (ResNet-50 FPN): it proposes regions,
  then predicts a heatmap for each end.
- Anchors are made smaller and more elongated for thin probes, and the points
  are ordered left to right.
- Big (59M parameters) and slow.

**U-Net (`unet`).**
- Marks every pixel as probe or not (ResNet18 encoder, full-resolution output).
- Each connected group of probe pixels is a probe; PCA gives its axis.
- No boxes and no NMS, so the angle is precise. But touching probes merge, a
  shadowed middle can split one probe in two, and it is slow.

**Classic CV (`classic`).**
- HSV threshold on the probe green, then connected regions filtered by
  length and elongation; the smallest rotated rectangle gives the axis.
- Thresholds were found by grid search.
- Very fast and precise when it finds a probe, but it misses about half of
  them (shade, glare, small probes).

**LSD lines (`lines`).**
- OpenCV's Line Segment Detector, keeping segments that lie on green pixels
  and joining them into one axis.
- The weakest method: the probe's edges and thin tip break into short,
  uneven segments.

**Box detectors (`detect`, `baseline`).**
- YOLOv8n with upright boxes, so they give no angle.
- They are the reference for detection, and `detect` is the first stage of
  `twostage`.

## Run it

All models run through one command, `erc_pose.infer`, with `--model <name>`.

### Live on the rover cameras

`scripts/field.sh CAMERA MODEL [MODEL ...]` is the easiest way:

```bash
scripts/field.sh mast obb                      # one model
scripts/field.sh mast obb pose unet            # several models: n / p switch between them
TILE=1 scripts/field.sh mast obb pose          # several models side by side, all on every frame
scripts/field.sh front pose                    # front-bottom camera (upside down, rotated 180° for you)
scripts/field.sh /dev/video0 obb               # any other camera
```

| camera | device | note |
|---|---|---|
| `mast` | `/dev/arducam-mast` | |
| `front` | `/dev/arducam-front-bottom` | mounted upside down, `--rotate 180` added |
| `rear` | `/dev/arducam-rear` | |

The camera names are udev aliases for `/dev/videoN`.

Keys:
- `q` or `Esc` quits.
- `n` / `p` switch to the next or previous model (without `TILE`).

Everything is kept in `field/<time>_<camera>/`:
- `detections.jsonl`, one line per frame and model;
- `frames/`, annotated frames that have a probe;
- `raw/`, every 10th clean frame, to label later.

With `TILE=1` every model runs on every frame, so the FPS is shared between them:
about 20 FPS for two YOLO models.

### Photos, videos, webcam

```bash
python -m erc_pose.infer --model obb --source path/to/photos --save out      # folder or single image
python -m erc_pose.infer --model pose --source video.mp4 --save out
python -m erc_pose.infer --model obb pose --tile --source 0 --show            # webcam, side by side
python -m erc_pose.infer --weights models/pose.pt --source 0 --show           # any weights file
```

| option | what it does |
|---|---|
| `--model NAME [NAME ...]` | model(s) from the table above |
| `--weights PATH` | instead of `--model`: Ultralytics weights or a spec like `unet:models/unet_v2.pt` |
| `--source` | image, folder, video, camera index or `/dev/...` |
| `--show` | live window |
| `--tile` | with several models: run all of them on each frame, side by side |
| `--rotate 180` | rotate camera frames (0/90/180/270) |
| `--conf 0.25` | detection threshold |
| `--save DIR` | annotated frames (`--save-detections-only` skips frames with no probe) |
| `--save-raw DIR` | every `--raw-every` (10) th clean frame |
| `--width/--height` | camera capture size, 960×600 by default (the training frame size) |
| `--device` | `cuda` or `cpu` (default: `cuda` when available) |

### Output

One JSON line per frame and model, on stdout:

```json
{"frame": 0, "model": "obb", "probes": [{"bbox": [x1, y1, x2, y2], "conf": 0.85,
 "kpts": [[x, y], [x, y]], "axis_deg": 12.3, "angle_deg": 12.3, "directed": false}]}
```

- `axis_deg` is the probe's axis in (-90°, 90°], measured in the image with y
  pointing down. **Use this value.**
- `kpts` are the two ends of the axis.
- `angle_deg` and `directed` matter only with a direction model
  (`--direction-weights`). The order-free pose model sets `directed: true`,
  but its point order means nothing.

### Check all models

```bash
python -m erc_pose.models                                           # list, weights present or missing
python -m erc_pose.models --check data/processed_v2/pose/images/test --out fps.json   # load each, measure FPS
```

## Data

| set | frames | probes | source |
|---|---|---|---|
| first labelled frames | 141 | 148 | HF dataset `erc-samples/erc-sample-pose`, `semi_supervised_dataset/` |
| pre-labelled, corrected in CVAT | 594 | 633 | `data/` of the same dataset; CVAT task 2686785 |
| close-ups (front-bottom and arm cameras, PNG) | 294 (21 without a probe) | 273 | `data/raw/closeup_20261108/` |
| drawn boxes for the first 141 frames | – | – | CVAT task 2660063, YOLO Detection export |

Together: 1029 frames from three cameras (mast, front-bottom, arm).

Labels: each probe is a `sample_pose` skeleton with two points, one at each end
of the probe. The point order does not matter. A `sample_box` rectangle is
optional; where it is missing, a box is built around the line (width 0.18 × length).

Splits are by recording session (camera + frames less than 60 s apart), never by
frame, because neighbouring frames are near-duplicates:
- **First 735 frames:** 515 / 110 / 110, made automatically.
- **Close-ups:** 201 / 47 / 46, split by hand in `data/raw/closeup_20261108/split.txt`.
- **Combined:** 716 train / 157 val / 156 test.

```bash
hf auth login
python -c "from huggingface_hub import snapshot_download; snapshot_download('erc-samples/erc-sample-pose', repo_type='dataset', local_dir='data/raw/erc-sample-pose')"

# first 735 frames: with drawn boxes (pose, detect) and with estimated widths (OBB, seg, U-Net)
python -m erc_pose.data.build_dataset \
    --yolo-pose data/raw/erc-sample-pose/semi_supervised_dataset path/to/task_2686785_export \
    --images data/raw/erc-sample-pose/data --boxes path/to/task_2660063_export \
    --min-line-px 3 --out data/processed_all
python -m erc_pose.data.build_dataset \
    --yolo-pose data/raw/erc-sample-pose/semi_supervised_dataset path/to/task_2686785_export \
    --images data/raw/erc-sample-pose/data --min-line-px 3 --out data/processed_all_est

# add the close-ups, keeping both splits as they are
python -m erc_pose.data.build_dataset --yolo-pose data/raw/closeup_20261108 --min-line-px 3 \
    --out data/processed_closeup
python scripts/add_closeup.py data/processed_all data/processed_closeup \
    data/raw/closeup_20261108/split.txt data/processed_v2
python scripts/add_closeup.py data/processed_all_est data/processed_closeup \
    data/raw/closeup_20261108/split.txt data/processed_v2_est
```

`add_closeup.py` also writes the test subsets `*_testold` (old frames: 110
frames, 153 probes) and `*_testnew` (close-ups: 46 frames, 41 probes).

## Train

```bash
scripts/run_v2.sh             # every YOLO model + Keypoint R-CNN, then evaluates each
scripts/run_v2_extra.sh       # U-Net and two-stage
scripts/run_cv_v2.sh          # cross-validation, pose vs OBB
```

Long runs: start them in their own systemd unit, so they survive the editor
or terminal closing:
`systemd-run --user --unit=train --same-dir --collect scripts/run_v2.sh`, then
follow with `journalctl --user -u train -f`.

One model at a time:

```bash
# YOLO: pose / pose11 / pose11s on data/processed_v2/pose, obb / obb11 on .._est/obb, seg / seg11 on .._est/seg
python -m erc_pose.train configs/pose_sym.yaml --data data/processed_v2/pose/data.yaml --set name=pose_v2
python -m erc_pose.train configs/obb.yaml --data data/processed_v2_est/obb/data.yaml --set name=obb_v2
python -m erc_pose.train configs/seg.yaml --data data/processed_v2_est/seg/data.yaml --set name=seg_v2
python -m erc_pose.train configs/detect.yaml --data data/processed_v2/detect/data.yaml --set name=detect_v2

python -m erc_pose.rcnn train --data data/processed_v2/pose --out runs/rcnn_v2
python -m erc_pose.semseg train --data data/processed_v2_est/obb --out runs/unet_v2
python -m erc_pose.twostage train --data data/processed_v2/pose --out runs/angle_v2
python -m erc_pose.classic --gt data/processed_v2/pose --name classic_v2    # tunes thresholds
python -m erc_pose.lines --gt data/processed_v2/pose --name lines_v2
```

Weights go to `runs/<name>/`. To make a model available by name, copy them
to `models/`:
- pose: `weights/last.pt`;
- OBB, seg and detect: `weights/best.pt`.

Training notes:
- **Augmentation** is on the fly: brightness, colour, zoom/crop, mosaic,
  flips, noise and blur.
- **No rotation for pose and detect:** rotation inflates the upright boxes of
  thin objects. OBB and seg use 180° rotation.
- **Pose** trains all 100 epochs with no early stopping (`configs/pose_sym.yaml`).
- `configs/experiments/` holds configs from earlier experiments.

## Evaluate

```bash
python -m erc_pose.evaluate --weights models/obb_v2.pt --data data/processed_v2_est/obb/data.yaml \
    --gt data/processed_v2/pose --name obb_v2 --out results/v2
python -m erc_pose.report results/v2 --parts     # table of every model, old frames vs close-ups
python -m erc_pose.crossval ...                  # cross-validation over sessions, see scripts/run_cv_v2.sh
```

`evaluate` writes `results/<name>.json`:
- recall and precision (a match is box IoU ≥ 0.5);
- axis error against the labelled line: mean, median and share within 5°;
- mAP and FPS.

Compare models on the axis error, not on box mAP: the label boxes of OBB and
seg come from the estimated width.

Each row of the test table is one run on ~200 probes, so differences of a few
tenths of a degree are noise. For a real comparison, use cross-validation.

## Results

Full comparison, test split of 156 frames / 194 probes. The "close-up" column
is the 41 close-up probes. All models are trained on the same 716 frames.

| model | params | recall | precision | axis error (median) | ≤5° | close-up axis error | FPS GPU / CPU |
|---|---|---|---|---|---|---|---|
| **YOLOv8n-OBB** | 3.1M | 0.89 | 0.86 | **2.1° (1.4°)** | **91%** | **1.5°** | 161 / 16 |
| **YOLOv8n-pose** | 3.1M | 0.88 | 0.85 | 2.2° (1.3°) | 88% | 1.7° | 176 / 15 |
| YOLO11n-OBB | 2.7M | 0.88 | 0.90 | 2.1° (1.5°) | 91% | 1.7° | 171 / 20 |
| YOLO11n-pose | 2.7M | 0.88 | 0.86 | 2.6° (2.0°) | 87% | 1.9° | 179 / 20 |
| YOLO11s-pose | 9.7M | **0.93** | 0.87 | 2.7° (1.6°) | 87% | 2.4° | 82 / 7 |
| U-Net (ResNet18 encoder) | 14.4M | 0.86 | 0.87 | 2.3° (1.4°) | 90% | 1.7° | 24 / 1.5 |
| two-stage: YOLOv8n + ResNet18 | 14.3M | 0.91 | 0.86 | 2.5° (1.6°) | 87% | 2.4° | 119 / 16 |
| YOLOv8n-seg | 3.4M | 0.91 | 0.82 | 3.2° (2.2°) | 85% | 2.2° | 128 / 13 |
| YOLO11n-seg | 2.9M | 0.91 | 0.83 | 3.5° (2.4°) | 83% | 2.3° | 136 / 14 |
| Keypoint R-CNN (ResNet-50 FPN) | 59M | 0.91 | 0.60 | 3.8° (1.6°) | 88% | 6.7° | 12 / 0.9 |
| classic CV, no learning | – | 0.53 | 0.83 | 2.5° (1.9°) | 89% | 1.7° | 949 (CPU) |
| LSD lines, no learning | – | 0.46 | 0.80 | 4.0° (2.8°) | 77% | 4.4° | 28 (CPU) |
| YOLOv8n detect | 3.1M | 0.91 | 0.86 | – | – | – | 174 / 18 |
| YOLOv8n detect, no augmentation | 3.1M | 0.89 | 0.86 | – | – | – | 172 / 18 |

FPS is measured end to end, one frame at a time at 960 px, on an RTX 4050
laptop GPU and a 20-thread CPU.

Close-up test frames only (46 frames, 41 probes):

| model | recall | axis error | within 5° |
|---|---|---|---|
| YOLOv8n-OBB | 0.90 | 1.5° | 97% |
| YOLOv8n-pose | 0.95 | 1.7° | 92% |
| YOLO11n-OBB | 0.90 | 1.7° | 95% |
| YOLO11n-pose | 0.90 | 1.9° | 92% |
| YOLO11s-pose | 0.95 | 2.4° | 92% |
| U-Net | 0.88 | 1.7° | 97% |
| two-stage | 0.95 | 2.4° | 87% |
| YOLOv8n-seg / YOLO11n-seg | 0.93 | 2.2° / 2.3° | 95% / 95% |
| Keypoint R-CNN | 0.93 | 6.7° | 84% |
| classic CV | 0.68 | 1.7° | 96% |
| LSD lines | 0.76 | 4.4° | 65% |
| earlier pose (no close-ups in training) | 0.93 | 3.3° | 84% |
| earlier OBB (no close-ups in training) | 0.90 | 1.9° | 97% |

Cross-validation over recording sessions on all 1029 frames (3 of 5 folds,
`scripts/run_cv_v2.sh`):

| | axis error | within 5° | recall | mAP50 |
|---|---|---|---|---|
| YOLOv8n-pose | 2.41° ± 0.19 | 91% | 0.945 ± 0.009 | 0.95 |
| YOLOv8n-OBB | 1.61° ± 0.02 | 95% | 0.934 ± 0.017 | 0.95 |

What the comparison shows:
- **OBB gives the most precise axis.** In cross-validation its lead over pose
  is several times the spread between folds.
- **The close-ups helped the pose model most.** Its test error fell from 3.2° to
  2.2° (3.3° → 1.7° on close probes). Recall on the old frames fell from 0.91 to
  0.86: in a few sunlit frames of one session the box stops before the thin tip,
  and IoU drops just under 0.5.
- **YOLO11 is not better than YOLOv8 here.** The bigger YOLO11s finds the most
  probes but runs at half the speed.
- **Segmentation finds probes well,** but the angle from its coarse mask is less
  precise.
- **U-Net gets close to the YOLO models** without any boxes, but at 24 FPS.
- **Two-stage keeps the detector's recall,** but its crop-based angle does not
  beat pose or OBB, which see the whole probe.
- **Keypoint R-CNN is 19× bigger and 15× slower,** with more false hits and
  large angle outliers on close probes (heatmap resolution; the left-to-right
  point order flips for near-vertical probes).
- **The classic methods find only half the probes.** When they find one, the
  angle is fine.

How the pose model got here (5-fold cross-validation on the first 141 frames):

| | axis error | within 5° |
|---|---|---|
| YOLOv8n-pose, standard loss | 17.8° ± 5.9 | 24% |
| keypoint loss without point order | 5.3° ± 2.2 | 69% |
| + drawn boxes | 4.7° ± 1.1 | 66% |
| + pseudo-labels, last epoch | 3.9° ± 1.4 | 78% |
| + 594 corrected frames (735 in all), test split | 3.2° | 82% |
| + 294 close-up frames (1029 in all), test split | 2.2° | 88% |

Known weak spots: probes under ~15 px, the thin tip of very close probes, and
low-confidence false hits on lens glare and screens. Thin probes are
sometimes reported twice, as two slightly shifted boxes.

Presentation material:
- `presentation/compare_chart.png`: axis error against FPS, made by
  `presentation/compare_chart.py`.
- `presentation/compare_grid.jpg`: the same test frames through several models,
  made by `scripts/compare_grid.py`.

## Pre-labelling new frames

```bash
python -m erc_pose.data.prelabel --weights models/pose_v2.pt \
    --images path/to/new/frames --labelled data/raw/erc-sample-pose/semi_supervised_dataset \
    --out data/cvat_prelabels
```

Produces a zip that CVAT imports as "Ultralytics YOLO Pose 1.0". Frames saved by
`field.sh` in `field/*/raw/` are good candidates: missed probes are the most
useful ones to label.
