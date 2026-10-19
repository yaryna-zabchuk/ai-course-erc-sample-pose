"""Reproducible train/val/test split that keeps a recording session together.

Consecutive frames of one recording are near-duplicates, so splitting them
at random leaks the test set into training.
"""

import random
import re
import warnings
from collections import defaultdict
from datetime import datetime

SPLITS = ("train", "val", "test")

# Roboflow renames "frame_0001.jpg" to "frame_0001_jpg.rf.<hash>.jpg".
_ROBOFLOW_SUFFIX = re.compile(r"(_(jpe?g|png|bmp))?\.rf\.[0-9a-zA-Z]+$")
DEFAULT_GROUP_REGEX = r"^(.*?)[_-]?\d+$"
# "<camera>_<YYYYMMDD-HHMMSS-mmm>", as written by the rover's capture script
_TIMESTAMPED = re.compile(r"^(?P<source>.+)_(?P<time>\d{8}-\d{6}-\d{3})$")
DEFAULT_SESSION_GAP_S = 60.0


def frame_time(stem):
    """(source, unix time) of a timestamped frame name, else None."""
    match = _TIMESTAMPED.match(_ROBOFLOW_SUFFIX.sub("", stem))
    if not match:
        return None
    return match["source"], datetime.strptime(match["time"], "%Y%m%d-%H%M%S-%f").timestamp()


def time_sessions(stems, gap_s=DEFAULT_SESSION_GAP_S):
    """Group timestamped frames: same source, less than gap_s seconds apart.

    Returns {stem: session name} for the stems that carry a timestamp.
    """
    by_source = defaultdict(list)
    for stem in stems:
        parsed = frame_time(stem)
        if parsed:
            by_source[parsed[0]].append((parsed[1], stem))
    sessions = {}
    for source, frames in by_source.items():
        frames.sort()
        index, previous = 0, None
        for time, stem in frames:
            if previous is not None and time - previous >= gap_s:
                index += 1
            sessions[stem] = f"{source}#{index}"
            previous = time
    return sessions


def group_key(stem, pattern=DEFAULT_GROUP_REGEX):
    """Session name of a frame: its file stem without the trailing frame number."""
    stem = _ROBOFLOW_SUFFIX.sub("", stem)
    match = re.match(pattern, stem)
    return match.group(1) if match and match.group(1) else stem


def split_groups(stems, ratios=(0.70, 0.15, 0.15), seed=0, pattern=DEFAULT_GROUP_REGEX,
                 session_gap=DEFAULT_SESSION_GAP_S, fold=None, folds=5):
    """Assign every stem to a split. Returns {stem: split_name}.

    Timestamped frames are grouped by time_sessions(), the rest by group_key().
    With `fold` set, the sessions are dealt into `folds` equal parts instead:
    part `fold` is the test split, the next part the val split and the rest train.
    """
    sessions = time_sessions(stems, session_gap)
    groups = defaultdict(list)
    for stem in sorted(stems):
        groups[sessions.get(stem) or group_key(stem, pattern)].append(stem)
    if len(groups) < (folds if fold is not None else len(SPLITS)):
        warnings.warn(
            f"only {len(groups)} recording session(s) found; splitting frame by frame, "
            "so near-duplicate frames may end up in different splits"
        )
        groups = {stem: [stem] for stem in sorted(stems)}

    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))  # stable: ties keep the shuffled order
    total = sum(len(members) for members in groups.values())

    if fold is None:
        bins = [ratio * total for ratio in ratios]
        split_of_bin = list(SPLITS)
    else:
        bins = [total / folds] * folds
        split_of_bin = ["train"] * folds
        split_of_bin[fold % folds] = "test"
        split_of_bin[(fold + 1) % folds] = "val"

    counts = [0] * len(bins)
    assignment = {}
    for name in names:
        # the bin that is furthest below its target takes the next session
        i = max(range(len(bins)), key=lambda k: bins[k] - counts[k])
        counts[i] += len(groups[name])
        for stem in groups[name]:
            assignment[stem] = split_of_bin[i]
    return assignment
