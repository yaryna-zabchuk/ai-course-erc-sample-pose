#!/usr/bin/env bash
# cross-validation, pose vs OBB, on all labelled frames (old + close-ups); 3 of the 5 session folds
# pose gets the drawn boxes, OBB the estimated widths (the drawn widths do not help its axis)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
if [ -n "${WAIT_PID:-}" ]; then tail --pid="$WAIT_PID" -f /dev/null; fi
SOURCES=(--export data/raw/erc-sample-pose/semi_supervised_dataset
         data/task_2686785_annotations_2026_10_21_00_06_15_ultralytics_yolo_pose data/raw/closeup_20261108
         --images data/raw/erc-sample-pose/data --min-line-px 3 --only-folds 0 1 2
         --weights-file last.pt --set patience=100)
echo "$(date +%T) cv pose"
$PY -m erc_pose.crossval "${SOURCES[@]}" --configs pose_sym --name v2_pose \
    --boxes "data/task_2660063_annotations_2026_10_21_00_22_14_ultralytics_yolo_detection (3)" > logs/cv_v2_pose.log 2>&1
echo "$(date +%T) cv obb"
$PY -m erc_pose.crossval "${SOURCES[@]}" --configs obb --name v2_obb > logs/cv_v2_obb.log 2>&1
echo "$(date +%T) cv done"
