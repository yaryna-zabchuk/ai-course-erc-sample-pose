#!/usr/bin/env bash
# non-YOLO models on data/processed_v2: U-Net segmentation and detector + angle regressor
# (the LSD baseline runs on the CPU: python -m erc_pose.lines)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/v2
PARTS=(--extra-gt)
while [ -n "${WAIT_UNIT:-}" ] && systemctl --user is-active --quiet "$WAIT_UNIT"; do sleep 30; done

echo "$(date +%T) train unet"
$PY -m erc_pose.semseg train --data data/processed_v2_est/obb --out runs/unet_v2 > logs/train_unet_v2.log 2>&1
echo "$(date +%T) eval unet"
$PY -m erc_pose.semseg evaluate --weights runs/unet_v2/last.pt --gt data/processed_v2/pose --name unet_v2 --out $OUT \
    "${PARTS[@]}" unet_v2_testold=data/processed_v2_testold/pose unet_v2_testnew=data/processed_v2_testnew/pose \
    > logs/eval_unet_v2.log 2>&1

echo "$(date +%T) train angle regressor"
$PY -m erc_pose.twostage train --data data/processed_v2/pose --out runs/angle_v2 > logs/train_angle_v2.log 2>&1
echo "$(date +%T) eval two-stage"
$PY -m erc_pose.twostage evaluate --detector runs/detect_v2/weights/best.pt --weights runs/angle_v2/last.pt \
    --gt data/processed_v2/pose --name twostage_v2 --out $OUT \
    "${PARTS[@]}" twostage_v2_testold=data/processed_v2_testold/pose twostage_v2_testnew=data/processed_v2_testnew/pose \
    > logs/eval_twostage_v2.log 2>&1
echo "$(date +%T) done"
