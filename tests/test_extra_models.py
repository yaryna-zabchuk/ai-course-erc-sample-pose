import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _bar(theta, length=120, width=10, centre=(200, 150), shape=(300, 400)):
    """Image with one green bar at theta degrees (y down)."""
    image = np.full((*shape, 3), (60, 80, 90), np.uint8)
    d = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    c = np.array(centre, dtype=float)
    p1, p2 = (c - d * length / 2).astype(int), (c + d * length / 2).astype(int)
    cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), (40, 200, 40), width)
    return image


@pytest.mark.parametrize("theta", [0, 25, 70, -40])
def test_classic_and_lines_find_a_green_bar(theta):
    from erc_pose import classic, lines
    from erc_pose.orientation import angular_error

    image = _bar(theta)
    for detect in (classic.detect, lines.detect):
        probes = detect(image)
        assert len(probes) == 1
        assert angular_error(probes[0].angle_deg, theta, 180.0) < 3


def test_blobs_to_probes_axis():
    from erc_pose.semseg import blobs_to_probes

    prob = np.zeros((300, 400), np.float32)
    cv2.line(prob, (100, 200), (300, 100), 1.0, 6)
    probes = blobs_to_probes(prob)
    assert len(probes) == 1
    assert probes[0].angle_deg % 180 == pytest.approx(np.degrees(np.arctan2(-100, 200)) % 180, abs=1)


def test_unet_keeps_the_frame_size():
    from erc_pose.semseg import UNet

    out = UNet(pretrained=False).eval()(torch.zeros(1, 3, 600, 960))
    assert out.shape == (1, 1, 600, 960)


def test_crop_rotation_turns_the_axis():
    from erc_pose.twostage import crop

    image = _bar(30, centre=(200, 150))
    patch = crop(image, (200, 150), 160, rotate_deg=45)
    ys, xs = np.nonzero(patch[..., 1] > 150)
    pts = np.c_[xs, ys].astype(float)
    v = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, -1]
    assert np.degrees(np.arctan2(v[1], v[0])) % 180 == pytest.approx((30 - 45) % 180, abs=2)


def test_regressor_target_is_end_free():
    from erc_pose.twostage import to_angle

    t = np.radians(2 * np.array([30.0, 30.0 + 180.0, -80.0]))
    angles = to_angle(np.c_[np.cos(t), np.sin(t)])
    assert angles == pytest.approx([30, 30, -80])
