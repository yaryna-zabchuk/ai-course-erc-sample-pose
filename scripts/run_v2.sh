#!/usr/bin/env bash
# train and evaluate the model comparison on data/processed_v2 (old frames + close-ups)
# usage: setsid nohup scripts/run_v2.sh [name ...] > logs/run_v2.log 2>&1 &
# WAIT_PID=<pid> waits for a running training to finish first
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/v2
mkdir -p logs $OUT

# name config dataset weights-file
JOBS=(
  "pose_v2 pose_sym data/processed_v2/pose last.pt"
  "pose11_v2 pose11 data/processed_v2/pose last.pt"
  "obb_v2 obb data/processed_v2_est/obb best.pt"
  "obb11_v2 obb11 data/processed_v2_est/obb best.pt"
  "seg_v2 seg data/processed_v2_est/seg best.pt"
  "seg11_v2 seg11 data/processed_v2_est/seg best.pt"
  "rcnn_v2 rcnn data/processed_v2/pose last.pt"
  "pose11s_v2 pose11s data/processed_v2/pose last.pt"
  "detect_v2 detect data/processed_v2/detect best.pt"
  "baseline_v2 baseline data/processed_v2/detect best.pt"
)

evaluate() {  # name weights dataset
  local name=$1 weights=$2 data=$3 task=${3##*/} root=${3%/*}
  $PY -m erc_pose.evaluate --weights "$weights" --data "$data/data.yaml" --gt "$root/pose" \
      --name "$name" --out $OUT > logs/eval_$name.log 2>&1
  for part in testold testnew; do
    $PY -m erc_pose.evaluate --weights "$weights" --data "${root}_$part/$task/data.yaml" \
        --gt "${root}_$part/pose" --name "${name}_$part" --out $OUT --no-fps > logs/eval_${name}_$part.log 2>&1
  done
}

if [ -n "${WAIT_PID:-}" ]; then tail --pid="$WAIT_PID" -f /dev/null; fi

rcnn() {  # train and evaluate Keypoint R-CNN
  [ -f runs/rcnn_v2/last.pt ] && grep -q "epoch 24/24" runs/rcnn_v2/train.log 2>/dev/null ||
    $PY -m erc_pose.rcnn train --data data/processed_v2/pose --out runs/rcnn_v2 > logs/train_rcnn_v2.log 2>&1
  $PY -m erc_pose.rcnn evaluate --weights runs/rcnn_v2/last.pt --gt data/processed_v2/pose --name rcnn_v2 \
      --extra-gt rcnn_v2_testold=data/processed_v2_testold/pose rcnn_v2_testnew=data/processed_v2_testnew/pose \
      --out $OUT > logs/eval_rcnn_v2.log 2>&1
}

for job in "${JOBS[@]}"; do
  read -r name config data weights <<< "$job"
  if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then continue; fi
  echo "$(date +%T) train $name"
  if [ "$name" = rcnn_v2 ]; then rcnn; continue; fi
  if [ ! -f runs/$name/weights/$weights ]; then
    $PY -m erc_pose.train configs/$config.yaml --data $data/data.yaml \
        --set name=$name exist_ok=true > logs/train_$name.log 2>&1 || { echo "train $name failed"; continue; }
  fi
  echo "$(date +%T) eval $name"
  evaluate $name runs/$name/weights/$weights $data
done
echo "$(date +%T) done"
