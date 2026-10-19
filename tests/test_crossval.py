from erc_pose.crossval import filtered_extra, summarise


def _touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_filtered_extra_drops_frames_near_val_and_test(tmp_path):
    fold = tmp_path / "fold" / "pose"
    _touch(fold / "images" / "test" / "mast_20260924-183600-000.jpg")
    _touch(fold / "images" / "train" / "mast_20260924-190000-000.jpg")
    extra = tmp_path / "extra"
    for stem in ("mast_20260924-183630-000",   # 30 s after a test frame: dropped
                 "mast_20260924-190010-000",   # near a train frame only: kept
                 "arm_20260924-183600-000"):   # other camera: kept
        _touch(extra / "images" / f"{stem}.jpg")
        _touch(extra / "labels" / f"{stem}.txt")
    out, kept = filtered_extra(extra, fold, tmp_path / "out", gap_s=60)
    assert kept == 2
    assert sorted(p.stem for p in (out / "images").iterdir()) == ["arm_20260924-183600-000", "mast_20260924-190010-000"]


def test_summarise_skips_non_numeric_and_nan():
    stats = summarise([{"a": 1.0, "b": "x", "c": float("nan")}, {"a": 3.0, "c": 2.0}])
    assert stats["a"] == {"mean": 2.0, "std": 1.0, "n": 2}
    assert stats["c"]["n"] == 1 and "b" not in stats
