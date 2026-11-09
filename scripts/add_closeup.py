"""Add a separately split set of frames to a built dataset, keeping the old split.

    python scripts/add_closeup.py data/processed_all data/processed_closeup \
        data/raw/closeup_20261108/split.txt data/processed_v2

split.txt has one "<stem> <train|val|test>" line per frame. Also writes a seg dataset
(the OBB polygons as masks) and two test-only datasets, <out>_testold and <out>_testnew,
for reporting the old and the new frames separately.
"""

import os
import sys
from pathlib import Path

TASKS = ("detect", "pose", "obb")
SPLITS = ("train", "val", "test")


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main(old, new, split_file, out):
    old, new, out = Path(old), Path(new), Path(out)
    split = dict(line.split() for line in Path(split_file).read_text().splitlines() if line.strip())
    for task in TASKS:
        for path in (old / task).glob("*/*/*"):
            link(path, out / task / path.relative_to(old / task))
        for path in (new / task).glob("*/*/*"):
            link(path, out / task / path.parts[-3] / split[path.stem] / path.name)
        (out / task / "data.yaml").write_text((old / task / "data.yaml").read_text())

    # segmentation: an OBB label line (class + 4 corners) is also a valid polygon label
    for path in (out / "obb").glob("*/*/*"):
        link(path, out / "seg" / path.relative_to(out / "obb"))
    (out / "seg" / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\n"
                                           "names:\n  0: probe\n")

    new_stems = set(split)
    for suffix, keep in (("testold", lambda s: s not in new_stems), ("testnew", lambda s: s in new_stems)):
        sub = Path(f"{out}_{suffix}")
        for task in TASKS + ("seg",):
            for kind in ("images", "labels"):
                for path in (out / task / kind / "test").iterdir():
                    if keep(path.stem):
                        link(path, sub / task / kind / "test" / path.name)
            (sub / task / "data.yaml").write_text((out / task / "data.yaml").read_text())

    for s in SPLITS:
        print(s, len(list((out / "pose" / "images" / s).iterdir())))


if __name__ == "__main__":
    main(*sys.argv[1:5])
