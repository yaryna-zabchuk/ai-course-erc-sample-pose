"""Label geometry: oriented boxes derived from the two annotated keypoints."""

import numpy as np


def probe_width(p1, p2, bbox_wh, min_ratio=0.05, min_px=2.0):
    """Estimate the probe's width from its axis and its axis-aligned box.

    A rectangle of length L and width w rotated by theta has an axis-aligned
    box of (L|cos| + w|sin|, L|sin| + w|cos|), so bw + bh = (L + w)(|cos| + |sin|).
    """
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    length = float(np.hypot(*v))
    if length == 0:
        return float(min(bbox_wh))
    c, s = abs(v[0]) / length, abs(v[1]) / length
    width = (bbox_wh[0] + bbox_wh[1]) / (c + s) - length
    return float(max(width, min_ratio * length, min_px))


def obb_from_keypoints(p1, p2, bbox_wh=None, width=None):
    """Four corners (4, 2) of the oriented box whose long axis runs p1 -> p2.

    The width is given, or else estimated from the axis-aligned box size.
    """
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    v = p2 - p1
    length = float(np.hypot(*v))
    if length == 0:
        raise ValueError("keypoints coincide, orientation is undefined")
    if width is None:
        width = probe_width(p1, p2, bbox_wh)
    normal = np.array([-v[1], v[0]]) / length * width / 2.0
    return np.stack([p1 + normal, p2 + normal, p2 - normal, p1 - normal])


def xywh_to_corners(bbox):
    """COCO (x, y, w, h) box as four corners (4, 2)."""
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def is_keypoint_extent(bbox, p1, p2, tol=1.0):
    """True when an (x, y, w, h) box is only the extent of the two keypoints.

    That is the case when a probe is annotated as its axis (a two-point
    skeleton in CVAT): the exported box is the extent of the line.
    """
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    x, y, w, h = bbox
    return bool(np.allclose([x, y, x + w, y + h], [*lo, *hi], atol=tol))


def bbox_from_keypoints(p1, p2, width_ratio):
    """(x, y, w, h) box around the axis p1 -> p2, assuming a probe width of `width_ratio` x its length."""
    length = float(np.hypot(*(np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float))))
    corners = obb_from_keypoints(p1, p2, width=max(width_ratio * length, 2.0))
    (x1, y1), (x2, y2) = corners.min(0), corners.max(0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
