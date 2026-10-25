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

The trained model is in the repo: `models/pose.pt`.

## Run it

```bash
# folder of photos or a video, annotated frames saved to out/
python -m erc_pose.infer --weights models/pose.pt --source path/to/photos --save out

# webcam (Esc or q to quit)
python -m erc_pose.infer --weights models/pose.pt --source 0 --show

# rover cameras (Linux); the front-bottom camera is mounted upside down
python -m erc_pose.infer --weights models/pose.pt --source /dev/arducam-mast --show
python -m erc_pose.infer --weights models/pose.pt --source /dev/arducam-front-bottom --rotate 180 --show
```

Every frame prints one JSON line, e.g.
`{"frame": 0, "probes": [{"bbox": [...], "conf": 0.85, "kpts": [...], "angle_deg": 12.3}]}`.
Cameras are captured at 960×600, the size of the training frames. `--save-raw`
keeps clean frames for labelling, `--conf` sets the detection threshold (0.25).

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
python -m erc_pose.train configs/pose_sym.yaml --data data/processed_all/pose/data.yaml --set name=pose_all
```

`configs/pose_sym.yaml` uses a keypoint loss that ignores which end of the
probe comes first (`erc_pose/symmetric.py`), trains all 100 epochs and the
weights to use are `weights/last.pt` (Ultralytics' `best.pt` can be an early
epoch whose points have not reached the probe ends yet). Augmentation is done on
the fly: brightness, colour, zoom/crop, mosaic, flips, noise, blur. No rotation,
it inflates the boxes of thin objects.

## Evaluate and benchmark

```bash
python -m erc_pose.evaluate --weights models/pose.pt --data data/processed_all/pose/data.yaml --name pose
python -m erc_pose.benchmark --weights models/pose.pt --images data/processed_all/pose/images/test
python -m erc_pose.report results
```

`evaluate` writes `results/<name>.json`: mAP50 / mAP50-95, recall, precision,
axis error in degrees (mean, median, share within 5°) and FPS on GPU and CPU.
`erc_pose.crossval` runs 5-fold cross-validation over recording sessions.

## Results

Test split of the 735 frames (110 frames, 153 probes):

| | axis error, mean / median | within 5° | recall | precision | FPS GPU / CPU |
|---|---|---|---|---|---|
| `models/pose.pt` | 3.2° / 2.3° | 82% | 0.91 | 0.83 | 184 / 20 |

GPU: RTX 4050 laptop, CPU: 20 threads, 960 px, one frame at a time.

How the pose model got there (5-fold cross-validation on the first 141 frames):

| | axis error | within 5° |
|---|---|---|
| baseline: YOLOv8n-pose, standard loss | 17.8° ± 5.9 | 24% |
| keypoint loss without point order | 5.3° ± 2.2 | 69% |
| + drawn boxes | 4.7° ± 1.1 | 66% |
| + pseudo-labels, last epoch | 3.9° ± 1.4 | 78% |

Box baseline (YOLOv8n detection, no augmentation): mAP50 0.53, recall 0.39 on
the first 141 frames. On the test frames that neither model saw in training,
the model trained on all 735 frames beats the earlier one: 2.9° vs 4.0° axis
error, recall 0.88 vs 0.80.

Known weak spots: very close probes (the thin tip gets cut off), probes under
~15 px, and occasional false hits on lens glare and screens.

## Pre-labelling new frames

```bash
python -m erc_pose.data.prelabel --weights models/pose.pt \
    --images path/to/new/frames --labelled data/raw/erc-sample-pose/semi_supervised_dataset \
    --out data/cvat_prelabels
```

Produces a zip that CVAT imports as "Ultralytics YOLO Pose 1.0".
