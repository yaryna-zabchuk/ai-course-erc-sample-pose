"""Keypoint and orientation metrics that Ultralytics does not report."""

import numpy as np

from erc_pose.orientation import angular_error


def box_iou(a, b):
    """IoU matrix between xyxy boxes a (N, 4) and b (M, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(pred_boxes, pred_conf, gt_boxes, iou_thr=0.5):
    """Greedy one-to-one matching, highest confidence first.

    Returns a list of (pred_index, gt_index) pairs.
    """
    iou = box_iou(pred_boxes, gt_boxes)
    matches, taken = [], set()
    for p in np.argsort(-np.asarray(pred_conf, dtype=float)):
        candidates = [(iou[p, g], g) for g in range(iou.shape[1]) if g not in taken]
        if not candidates:
            break
        best_iou, g = max(candidates)
        if best_iou >= iou_thr:
            taken.add(g)
            matches.append((int(p), int(g)))
    return matches


def keypoint_mae(pred_kpts, gt_kpts):
    """Mean absolute error of keypoint coordinates in pixels, over (N, K, 2)."""
    pred_kpts = np.asarray(pred_kpts, dtype=float)
    gt_kpts = np.asarray(gt_kpts, dtype=float)
    if pred_kpts.size == 0:
        return float("nan")
    return float(np.abs(pred_kpts - gt_kpts).mean())


def angle_stats(pred_deg, gt_deg, period=360.0, tolerance=5.0):
    """Mean/median absolute angle error and the share of errors within tolerance."""
    if len(pred_deg) == 0:
        return {"mae_deg": float("nan"), "median_deg": float("nan"), "within_tol": float("nan")}
    err = angular_error(pred_deg, gt_deg, period)
    return {
        "mae_deg": float(err.mean()),
        "median_deg": float(np.median(err)),
        "within_tol": float((err <= tolerance).mean()),
    }
