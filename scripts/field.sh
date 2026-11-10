#!/usr/bin/env bash
# field test: run one or more models on a rover camera, show them live and keep a record
# usage: scripts/field.sh CAMERA MODEL [MODEL ...]
#   CAMERA  mast | front | rear | /dev/videoN | camera index (udev aliases from /etc/udev/rules.d)
#   MODEL   names from `python -m erc_pose.models` (pose, obb, seg, unet, twostage, rcnn, classic, ...)
# keys: n / p next / previous model, q or Esc quit
# saved to field/<time>_<camera>/: detections.jsonl (one line per frame), annotated frames
# with a probe, and every 10th clean frame (for labelling later)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CAMERA=${1:?camera: mast, front, rear or /dev/...}
shift
[ $# -gt 0 ] || set -- pose obb

ROTATE=0
case "$CAMERA" in
  mast) SOURCE=/dev/arducam-mast ;;
  front) SOURCE=/dev/arducam-front-bottom; ROTATE=180 ;;  # mounted upside down
  rear) SOURCE=/dev/arducam-rear ;;
  *) SOURCE=$CAMERA ;;
esac

OUT=field/$(date +%Y%m%d-%H%M%S)_$(basename "$CAMERA")
mkdir -p "$OUT"
echo "models: $*  camera: $SOURCE  saving to $OUT" >&2
$PY -m erc_pose.infer --model "$@" --source "$SOURCE" --rotate $ROTATE --show \
    --save "$OUT/frames" --save-detections-only --save-raw "$OUT/raw" --raw-every 10 \
    | tee "$OUT/detections.jsonl"
