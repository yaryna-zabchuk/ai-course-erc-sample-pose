from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from erc_pose.predict import probes_from_result  # noqa: E402


def _result(boxes, conf, kpts=None, obb=None):
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                              conf=torch.tensor(conf, dtype=torch.float32)),
        keypoints=None if kpts is None else SimpleNamespace(xy=torch.tensor(kpts, dtype=torch.float32)),
        obb=obb,
    )


def test_pose_result_with_no_detections():
    assert probes_from_result(_result([], [], kpts=np.zeros((0, 2, 2)))) == []


def test_pose_result_gives_directed_angle():
    (probe,) = probes_from_result(_result([[0, 0, 10, 10]], [0.9], kpts=[[[0, 0], [0, 10]]]))
    assert probe.angle_deg == pytest.approx(90) and probe.directed


def test_fuse_direction_turns_obb_axis_to_match_pose():
    from erc_pose.predict import Probe, fuse_direction

    box = np.array([0, 0, 10, 10], dtype=float)
    obb = [Probe(box, 0.9, None, 10.0, False), Probe(box + 100, 0.8, None, -45.0, False)]
    pose = [Probe(box, 0.7, np.zeros((2, 2)), -165.0, True)]
    fused = fuse_direction(obb, pose)
    assert fused[0].angle_deg == pytest.approx(-170) and fused[0].directed
    assert fused[1].angle_deg == pytest.approx(-45) and not fused[1].directed


def test_obb_axis_ends_follow_the_fused_direction():
    from erc_pose.orientation import angle_from_keypoints
    from erc_pose.predict import Probe, axis_ends, fuse_direction

    ends = axis_ends(np.array([50.0, 50.0, 4.0, 20.0, 0.0]), 90.0)  # tall box: long axis is vertical
    assert ends == pytest.approx(np.array([[50, 40], [50, 60]]))
    box = np.array([40, 40, 60, 60], dtype=float)
    obb = Probe(box, 0.9, ends, 90.0, False)
    (fused,) = fuse_direction([obb], [Probe(box, 0.8, None, -80.0, True)])
    assert fused.angle_deg == pytest.approx(-90)
    assert float(angle_from_keypoints(fused.kpts[0], fused.kpts[1])) == pytest.approx(-90)
    assert fused.agreement_deg == pytest.approx(10)


def test_unflip_restores_keypoints_and_angle():
    from erc_pose.predict import Probe, unflip

    probe = Probe(np.array([10.0, 20.0, 30.0, 60.0]), 0.9, np.array([[10.0, 20.0], [30.0, 60.0]]), 0.0, True)
    back = unflip(probe, -1, width=100, height=80)  # flipped both ways
    assert back.xyxy == pytest.approx(np.array([70, 20, 90, 60]))
    assert back.kpts == pytest.approx(np.array([[90, 60], [70, 20]]))
    assert back.angle_deg == pytest.approx(float(np.degrees(np.arctan2(-40, -20))))


def test_fuse_direction_votes_by_confidence():
    from erc_pose.predict import Probe, fuse_direction

    box = np.array([0, 0, 10, 10], dtype=float)
    obb = [Probe(box, 0.9, None, 0.0, False)]
    votes = [Probe(box, 0.9, None, 175.0, True), Probe(box, 0.3, None, 5.0, True), Probe(box, 0.3, None, 0.0, True)]
    assert fuse_direction(obb, votes)[0].angle_deg == pytest.approx(180)
