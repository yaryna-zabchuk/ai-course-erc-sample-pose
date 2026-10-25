#!/usr/bin/env bash
# whole pipeline: fix labels, build data, train, pseudo-label, retrain obb, evaluate
# usage: scripts/run_all.sh [pose export] [unlabelled frames]
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
# main model: obb with pseudo-labels
$PY -m erc_pose.evaluate --weights runs/obb_semi/weights/best.pt --data data/processed/obb/data.yaml \
    --name obb_semi > logs/eval_obb_semi.log 2>&1
# optional: direction from the pose model
$PY -m erc_pose.evaluate --weights runs/obb/weights/best.pt --data data/processed/obb/data.yaml \
    --direction-weights runs/pose/weights/best.pt --name obb+pose > logs/eval_obb+pose.log 2>&1
$PY -m erc_pose.evaluate --weights runs/obb_semi/weights/best.pt --data data/processed/obb/data.yaml \
    --direction-weights runs/pose/weights/best.pt --name obb_semi+pose > logs/eval_obb_semi+pose.log 2>&1
$PY -m erc_pose.report results
