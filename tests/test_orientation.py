import numpy as np
import pytest

from erc_pose.metrics import angle_stats, box_iou, keypoint_mae, match_boxes
from erc_pose.orientation import angle_from_keypoints, angle_from_xywhr, angular_error, orientation_vector, wrap_angle


@pytest.mark.parametrize("p2, expected", [
    ((1, 0), 0.0),
    ((0, 1), 90.0),      # image y points down
    ((-1, 0), 180.0),
    ((0, -1), -90.0),
    ((1, 1), 45.0),
])
def test_angle_from_keypoints(p2, expected):
    assert angle_from_keypoints((0, 0), p2) == pytest.approx(expected)


def test_angle_from_keypoints_batched():
    p1 = np.zeros((3, 2))
    p2 = np.array([[1, 0], [0, 1], [-1, -1]])
    assert angle_from_keypoints(p1, p2) == pytest.approx([0, 90, -135])


def test_wrap_angle_range():
    assert wrap_angle(190) == pytest.approx(-170)
    assert wrap_angle(-180) == pytest.approx(180)
    assert wrap_angle(95, period=180) == pytest.approx(-85)


def test_angular_error_wraps_around():
    assert angular_error(179, -179) == pytest.approx(2)
    assert angular_error(10, 190) == pytest.approx(180)
    assert angular_error(10, 190, period=180) == pytest.approx(0)
    assert angular_error(-88, 89, period=180) == pytest.approx(3)


def test_orientation_vector():
    assert orientation_vector((0, 0), (3, 4)) == pytest.approx([0.6, 0.8])
    assert orientation_vector((1, 1), (1, 1)) == pytest.approx([0, 0])


def test_angle_from_xywhr_uses_long_axis():
    assert angle_from_xywhr([0, 0, 10, 2, 0.0]) == pytest.approx(0)
    assert angle_from_xywhr([0, 0, 2, 10, 0.0]) == pytest.approx(90)
    assert angle_from_xywhr([0, 0, 2, 10, np.pi / 4]) == pytest.approx(-45)


def test_match_boxes_is_one_to_one():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[1, 1, 10, 10], [0, 0, 10, 10], [100, 100, 110, 110]]
    assert box_iou(pred, gt)[1, 0] == pytest.approx(1.0)
    assert match_boxes(pred, [0.9, 0.8, 0.7], gt) == [(0, 0)]
    assert match_boxes(np.zeros((0, 4)), [], gt) == []


def test_keypoint_mae_and_angle_stats():
    assert keypoint_mae([[[0, 0], [10, 10]]], [[[2, 0], [10, 12]]]) == pytest.approx(1.0)
    assert np.isnan(keypoint_mae([], []))
    stats = angle_stats([179, 0, 20], [-179, 4, 0])
    assert stats["mae_deg"] == pytest.approx((2 + 4 + 20) / 3)
    assert stats["within_tol"] == pytest.approx(2 / 3)
