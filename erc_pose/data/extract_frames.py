"""Extract frames from rover recordings, skipping near-duplicates.

    python -m erc_pose.data.extract_frames recordings/*.mp4 --out data/raw/frames

Frames are named `<video>_<frame index>.jpg`, which is what the split script
uses to keep one recording inside one split.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def extract(video, out_dir, every=10, min_diff=4.0):
    """Save every `every`-th frame that differs enough from the last saved one."""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open {video}")
    saved, last, index = 0, None, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            thumb = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if last is None or np.abs(thumb - last).mean() >= min_diff:
                cv2.imwrite(str(out_dir / f"{video.stem}_{index:06d}.jpg"), frame)
                last, saved = thumb, saved + 1
        index += 1
    capture.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out", default="data/raw/frames")
    parser.add_argument("--every", type=int, default=10, help="consider every N-th frame")
    parser.add_argument("--min-diff", type=float, default=4.0,
                        help="minimum mean absolute grey-level difference to the last saved frame")
    args = parser.parse_args()
    for video in args.videos:
        print(f"{video}: {extract(video, args.out, args.every, args.min_diff)} frames")


if __name__ == "__main__":
    main()
