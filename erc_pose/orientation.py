"""Probe angles in the image plane, degrees in (-180, 180], x right, y down."""

import numpy as np


def wrap_angle(angle, period=360.0):
    """Wrap angle(s) into (-period/2, period/2]."""
    half = period / 2.0
    wrapped = (np.asarray(angle, dtype=float) + half) % period - half
    return np.where(wrapped == -half, half, wrapped)


def orientation_vector(p1, p2):
    """Unit vector(s) from p1 to p2. Zero-length inputs give a zero vector."""
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def angle_from_keypoints(p1, p2):
    """theta = atan2(y2 - y1, x2 - x1) in degrees, for points of shape (..., 2)."""
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    return wrap_angle(np.degrees(np.arctan2(v[..., 1], v[..., 0])))


def angular_error(pred, gt, period=360.0):
    """Absolute circular difference; period=180 for an undirected axis."""
    return np.abs(wrap_angle(np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float), period))


def angle_from_xywhr(xywhr):
    """Long-axis angle of (cx, cy, w, h, r) boxes, in (-90, 90]."""
    xywhr = np.asarray(xywhr, dtype=float)
    w, h, r = xywhr[..., 2], xywhr[..., 3], xywhr[..., 4]
    return wrap_angle(np.degrees(r) + np.where(w >= h, 0.0, 90.0), period=180.0)
