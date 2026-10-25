import json
from collections import Counter

import numpy as np
import pytest
import yaml

from erc_pose.data.build_dataset import build, label_lines
from erc_pose.data.geometry import obb_from_keypoints, probe_width
from erc_pose.data.split import group_key, split_groups
from erc_pose.orientation import angle_from_keypoints


def _rotated_probe(length, width, theta_deg, centre=(100.0, 100.0)):
    """Keypoints and axis-aligned box size of a length x width rectangle."""
    t = np.radians(theta_deg)
    d = np.array([np.cos(t), np.sin(t)])
    p1, p2 = centre - d * length / 2, centre + d * length / 2
    bbox_wh = (length * abs(d[0]) + width * abs(d[1]), length * abs(d[1]) + width * abs(d[0]))
    return p1, p2, bbox_wh


@pytest.mark.parametrize("theta", [0, 30, 90, 135, -60])
def test_probe_width_is_recovered_from_bbox(theta):
    p1, p2, bbox_wh = _rotated_probe(80, 12, theta)
    assert probe_width(p1, p2, bbox_wh) == pytest.approx(12)


def test_obb_from_keypoints():
    p1, p2, bbox_wh = _rotated_probe(80, 12, 30)
    corners = obb_from_keypoints(p1, p2, bbox_wh)
    assert corners.mean(0) == pytest.approx([100, 100])
    assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(80)
    assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(12)
    assert float(angle_from_keypoints(corners[0], corners[1])) == pytest.approx(30)
    # the oriented box spans exactly the annotated axis-aligned box
    assert np.ptp(corners, axis=0) == pytest.approx(bbox_wh)


def test_group_key():
    assert group_key("run3_000120") == "run3"
    assert group_key("run3_000120_jpg.rf.ab12cd34") == "run3"
    assert group_key("IMG") == "IMG"


def test_split_keeps_sessions_together_and_is_reproducible():
    stems = [f"run{s}_{i:04d}" for s in range(20) for i in range(10)]
    first = split_groups(stems, seed=1)
    assert first == split_groups(stems, seed=1)
    for s in range(20):
        assert len({first[f"run{s}_{i:04d}"] for i in range(10)}) == 1
    assert Counter(first.values()) == {"train": 140, "val": 30, "test": 30}


def test_split_falls_back_to_frames_for_a_single_session():
    with pytest.warns(UserWarning):
        assignment = split_groups([f"run_{i:04d}" for i in range(100)])
    assert Counter(assignment.values()) == {"train": 70, "val": 15, "test": 15}


def test_label_lines():
    ann = {"bbox": [50, 100, 100, 20], "keypoints": [50, 110, 2, 150, 110, 2]}
    lines = label_lines([ann], 200, 200)
    assert lines["detect"] == ["0 0.500000 0.550000 0.500000 0.100000"]
    assert lines["pose"] == ["0 0.500000 0.550000 0.500000 0.100000 0.250000 0.550000 2 0.750000 0.550000 2"]
    obb = np.array(lines["obb"][0].split()[1:], dtype=float).reshape(4, 2) * 200
    assert obb.min(0) == pytest.approx([50, 100])
    assert obb.max(0) == pytest.approx([150, 120])


def _write_coco(folder, with_keypoints):
    images, annotations = [], []
    for i in range(12):
        name = f"run{i % 4}_{i:03d}.jpg"
        (folder / name).write_bytes(b"")
        images.append({"id": i, "file_name": name, "width": 200, "height": 200})
        ann = {"id": i, "image_id": i, "category_id": 1, "bbox": [50, 100, 100, 20]}
        if with_keypoints:
            ann["keypoints"] = [50, 110, 2, 150, 110, 2]
        annotations.append(ann)
    path = folder / "annotations.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}))
    return path


def test_build_writes_three_datasets(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    yamls = build([_write_coco(raw, True)], tmp_path / "out")
    assert set(yamls) == {"detect", "pose", "obb"}
    assert yaml.safe_load(yamls["pose"].read_text())["kpt_shape"] == [2, 3]
    for task in yamls:
        root = tmp_path / "out" / task
        images = sorted(p.stem for p in root.glob("images/*/*.jpg"))
        labels = sorted(p.stem for p in root.glob("labels/*/*.txt"))
        assert images == labels and len(images) == 12


def test_build_without_keypoints_is_detection_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    assert set(build([_write_coco(raw, False)], tmp_path / "out")) == {"detect"}


def test_yolo_pose_export_gives_the_same_labels_as_coco(tmp_path):
    from erc_pose.data.synthetic import generate

    coco = generate(tmp_path / "raw", count=12, sessions=4, width=160, height=120)
    build([coco], tmp_path / "from_coco")
    # the pose dataset has the layout of a CVAT "Ultralytics YOLO Pose" export
    build(out_dir=tmp_path / "from_yolo", yolo_dirs=[tmp_path / "from_coco" / "pose"])
    for task in ("detect", "pose", "obb"):
        expected = {p.name: p.read_text() for p in (tmp_path / "from_coco" / task).glob("labels/*/*.txt")}
        actual = {p.name: p.read_text() for p in (tmp_path / "from_yolo" / task).glob("labels/*/*.txt")}
        assert expected.keys() == actual.keys()
        for name, text in expected.items():
            exp = np.array(text.split(), dtype=float)
            assert np.array(actual[name].split(), dtype=float) == pytest.approx(exp, abs=2e-5), (task, name)


def test_time_sessions_split_on_gaps_per_camera():
    from erc_pose.data.split import time_sessions

    stems = ["mast_20260924-183600-000", "mast_20260924-183630-500", "mast_20260924-183900-000",
             "arm_20260924-183600-000", "other_frame_0001"]
    sessions = time_sessions(stems, gap_s=60)
    assert sessions[stems[0]] == sessions[stems[1]] != sessions[stems[2]]
    assert sessions[stems[3]] not in {sessions[stems[0]], sessions[stems[2]]}
    assert stems[4] not in sessions


def test_line_boxes_are_rebuilt_from_keypoints():
    from erc_pose.data.build_dataset import repair_line_boxes

    line = {"bbox": [100, 100, 0, 50], "keypoints": [100, 100, 2, 100, 150, 2]}
    real = {"bbox": [95, 100, 10, 50], "keypoints": [100, 100, 2, 100, 150, 2]}
    assert repair_line_boxes({"a": [line, real]}, width_ratio=0.2) == 1
    assert line["bbox"] == pytest.approx([95, 100, 10, 50])
    assert real["bbox"] == [95, 100, 10, 50]


def test_extra_train_goes_to_train_and_keeps_the_split(tmp_path):
    from erc_pose.data.synthetic import generate

    coco = generate(tmp_path / "raw", count=16, sessions=4, width=160, height=120)
    build([coco], tmp_path / "base")
    extra = generate(tmp_path / "extra_raw", count=4, sessions=1, width=160, height=120, seed=5)
    for path in (tmp_path / "extra_raw").glob("*.jpg"):  # distinct names from the base set
        path.rename(path.with_name("x" + path.name))
    import json as _json
    data = _json.loads(extra.read_text())
    for img in data["images"]:
        img["file_name"] = "x" + img["file_name"]
    extra.write_text(_json.dumps(data))
    build([extra], tmp_path / "extra_built")
    yolo_extra = tmp_path / "extra_yolo"
    for kind in ("images", "labels"):
        (yolo_extra / kind).mkdir(parents=True)
        for path in (tmp_path / "extra_built" / "pose" / kind).glob("*/*"):
            (yolo_extra / kind / path.name).write_bytes(path.read_bytes())

    build([coco], tmp_path / "combined", extra_train=[yolo_extra])
    split_of = lambda root: {p.stem: p.parent.name for p in (root / "pose" / "images").glob("*/*")}
    base, combined = split_of(tmp_path / "base"), split_of(tmp_path / "combined")
    assert {k: v for k, v in combined.items() if not k.startswith("x")} == base
    assert {v for k, v in combined.items() if k.startswith("x")} == {"train"}


def test_folds_cover_every_session_once_as_test():
    stems = [f"run{s}_{i:04d}" for s in range(10) for i in range(5)]
    tests = Counter()
    for fold in range(5):
        assignment = split_groups(stems, fold=fold, folds=5)
        assert Counter(assignment.values()) == {"train": 30, "val": 10, "test": 10}
        tests.update(s for s, split in assignment.items() if split == "test")
    assert set(tests) == set(stems) and set(tests.values()) == {1}


def test_left_to_right_order():
    from erc_pose.data.build_dataset import order_left_to_right

    right_first = np.array([[10.0, 5, 2], [2, 7, 2]])
    assert order_left_to_right(right_first)[0, 0] == 2
    vertical = np.array([[5.0, 9, 2], [5, 1, 2]])  # same x: upper end (smaller y) first
    assert order_left_to_right(vertical)[0, 1] == 1


def test_attach_boxes_gives_each_line_its_drawn_box():
    from erc_pose.data.build_dataset import attach_boxes

    images = {"f": {"width": 100, "height": 100}}
    line = {"bbox": [10, 10, 20, 0], "keypoints": [10, 10, 2, 30, 10, 2]}
    anns = {"f": [line]}
    boxes = {"f": [[0.2, 0.1, 0.3, 0.1],      # contains the line: x 5-35, y 5-15
                   [0.8, 0.8, 0.1, 0.1]]}     # elsewhere, no line -> box-only annotation
    assert attach_boxes(images, anns, boxes) == 1
    assert line["bbox"] == pytest.approx([5, 5, 30, 10])
    assert len(anns["f"]) == 2 and "keypoints" not in anns["f"][1]


def test_predicted_boxes_do_not_add_probes():
    from erc_pose.data.build_dataset import attach_boxes

    images = {"f": {"width": 100, "height": 100}}
    anns = {"f": [{"bbox": [10, 10, 20, 0], "keypoints": [10, 10, 2, 30, 10, 2]}]}
    boxes = {"f": [[0.2, 0.1, 0.3, 0.1], [0.8, 0.8, 0.1, 0.1]]}
    assert attach_boxes(images, anns, boxes, add_unmatched=False) == 1
    assert len(anns["f"]) == 1


def test_drop_short_lines():
    from erc_pose.data.build_dataset import drop_short_lines

    anns = {"f": [{"bbox": [0, 0, 1, 1], "keypoints": [10, 10, 2, 11, 10, 2]},
                  {"bbox": [0, 0, 40, 1], "keypoints": [10, 10, 2, 50, 10, 2]}]}
    assert drop_short_lines(anns, 3) == 1 and len(anns["f"]) == 1


def test_yolo_pose_export_keeps_frames_without_labels(tmp_path):
    import cv2
    from erc_pose.data.build_dataset import load_yolo_pose

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for stem in ("a", "b"):
        cv2.imwrite(str(tmp_path / "images" / f"{stem}.jpg"), np.zeros((20, 30, 3), np.uint8))
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.4 0.5 2 0.6 0.5 2\n")
    (tmp_path / "train.txt").write_text("images/a.jpg\nimages/b.jpg\n")
    images, anns = load_yolo_pose(tmp_path)
    assert set(images) == {"a", "b"} and len(anns["a"]) == 1 and not anns["b"]
