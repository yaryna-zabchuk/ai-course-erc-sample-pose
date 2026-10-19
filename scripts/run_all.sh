#!/usr/bin/env bash
# Full pipeline on the Hugging Face data: fix label directions, build the datasets, train the
# baseline / detect / pose / OBB models, pseudo-label the unlabelled frames, retrain OBB with them,
# evaluate everything on the test split and print the comparison table.
#   scripts/run_all.sh [YOLO Pose export] [folder with unlabelled frames]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
EXPORT=${1:-data/raw/erc-sample-pose/semi_supervised_dataset}
UNLABELLED=${2:-data/raw/erc-sample-pose/data}
FIXED=data/raw/semi_supervised_fixed
mkdir -p logs results

echo "== labels and datasets"
$PY -m erc_pose.data.check_direction "$EXPORT" --out "$FIXED" --report results/direction_report.csv
$PY -m erc_pose.data.build_dataset --yolo-pose "$FIXED" --out data/processed

for config in baseline detect pose obb; do
  task=$config; [ "$config" = baseline ] && task=detect
  echo "== $config"
  $PY -m erc_pose.train "configs/$config.yaml" --set exist_ok=true > "logs/train_$config.log" 2>&1
  $PY -m erc_pose.evaluate --weights "runs/$config/weights/best.pt" \
      --data "data/processed/$task/data.yaml" --name "$config" > "logs/eval_$config.log" 2>&1
done

echo "== semi-supervised OBB"
$PY -m erc_pose.data.pseudo_label --weights runs/pose/weights/best.pt --obb-weights runs/obb/weights/best.pt \
    --images "$UNLABELLED" --dataset data/processed/pose --out data/pseudo
$PY -m erc_pose.data.build_dataset --yolo-pose "$FIXED" --extra-train data/pseudo --out data/processed_semi
$PY -m erc_pose.train configs/obb.yaml --data data/processed_semi/obb/data.yaml \
    --set name=obb_semi exist_ok=true > logs/train_obb_semi.log 2>&1

echo "== evaluation"
# the deliverable: the semi-supervised OBB alone (axis only)
$PY -m erc_pose.evaluate --weights runs/obb_semi/weights/best.pt --data data/processed/obb/data.yaml \
    --name obb_semi > logs/eval_obb_semi.log 2>&1
# optional: OBB gives the axis, the pose model (human labels only) the direction
$PY -m erc_pose.evaluate --weights runs/obb/weights/best.pt --data data/processed/obb/data.yaml \
    --direction-weights runs/pose/weights/best.pt --name obb+pose > logs/eval_obb+pose.log 2>&1
$PY -m erc_pose.evaluate --weights runs/obb_semi/weights/best.pt --data data/processed/obb/data.yaml \
    --direction-weights runs/pose/weights/best.pt --name obb_semi+pose > logs/eval_obb_semi+pose.log 2>&1
$PY -m erc_pose.report results
