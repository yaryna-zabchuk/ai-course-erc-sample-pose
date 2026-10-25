#!/usr/bin/env bash
# build the dataset, train baseline / detect / pose, evaluate on the test split
# usage: scripts/run_all.sh LABELLED CORRECTED BOXES FRAMES
#   LABELLED   semi_supervised_dataset from the HF dataset (YOLO Pose export)
#   CORRECTED  CVAT export of the corrected frames (task 2686785, YOLO Pose)
#   BOXES      CVAT export of the drawn boxes (task 2660063, YOLO Detection)
#   FRAMES     data/ folder of the HF dataset (images of the corrected frames)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LABELLED=${1:?labelled export}
CORRECTED=${2:?corrected export}
BOXES=${3:?box export}
FRAMES=${4:?frames folder}
DATA=data/processed_all
mkdir -p logs results

$PY -m erc_pose.data.build_dataset --yolo-pose "$LABELLED" "$CORRECTED" --images "$FRAMES" \
    --boxes "$BOXES" --min-line-px 3 --out "$DATA"

for config in baseline detect; do
  $PY -m erc_pose.train configs/$config.yaml --data $DATA/detect/data.yaml \
      --set name=$config exist_ok=true > logs/train_$config.log 2>&1
  $PY -m erc_pose.evaluate --weights runs/$config/weights/best.pt --data $DATA/detect/data.yaml \
      --gt $DATA/detect --name $config > logs/eval_$config.log 2>&1
done

$PY -m erc_pose.train configs/pose_sym.yaml --data $DATA/pose/data.yaml \
    --set name=pose exist_ok=true > logs/train_pose.log 2>&1
$PY -m erc_pose.evaluate --weights runs/pose/weights/last.pt --data $DATA/pose/data.yaml \
    --name pose > logs/eval_pose.log 2>&1

$PY -m erc_pose.report results
