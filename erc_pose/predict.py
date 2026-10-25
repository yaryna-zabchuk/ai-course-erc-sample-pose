"""Turn an Ultralytics result into plain probe detections."""

from dataclasses import dataclass, replace

import numpy as np

from erc_pose.metrics import box_iou
from erc_pose.orientation import angle_from_keypoints, angle_from_xywhr, angular_error, wrap_angle


@dataclass
class Probe:
    xyxy: np.ndarray            # axis-aligned box in pixels
    conf: float
    kpts: np.ndarray | None     # (2, 2); for obb the axis ends
    angle_deg: float | None     # None for plain detection models
    directed: bool              # False when the angle is only known modulo 180
    agreement_deg: float | None = None  # fused: obb axis vs pose direction
    direction_conf: float | None = None  # classifier confidence, 0..1

    def to_message(self):
        """Compact dict for the operator link."""
        message = {"bbox": [round(float(v)) for v in self.xyxy], "conf": round(self.conf, 3)}
        if self.kpts is not None:
            message["kpts"] = [[round(float(x), 1), round(float(y), 1)] for x, y in self.kpts]
        if self.angle_deg is not None:
            message["angle_deg"] = round(self.angle_deg, 1)
            message["directed"] = self.directed
        return message


def axis_ends(xywhr, angle_deg):
    """Ends (2, 2) of an oriented box's long axis, ordered along angle_deg."""
    half = max(xywhr[2], xywhr[3]) / 2 * np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    return np.stack([xywhr[:2] - half, xywhr[:2] + half])


def probes_from_result(result):
    """List of Probe for one image, for detect, pose and obb models alike."""
    if result.obb is not None:
        boxes = result.obb.xyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        xywhr = result.obb.xywhr.cpu().numpy().reshape(-1, 5)
        angles = angle_from_xywhr(xywhr).reshape(-1)
        return [Probe(b, float(c), axis_ends(r, a), float(a), False)
                for b, c, r, a in zip(boxes, confs, xywhr, angles)]

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    if not len(boxes):
        return []
    if result.keypoints is None:
        return [Probe(b, float(c), None, None, False) for b, c in zip(boxes, confs)]
    kpts = result.keypoints.xy.cpu().numpy().reshape(len(boxes), -1, 2)
    return [Probe(b, float(c), k, float(angle_from_keypoints(k[0], k[1])), True)
            for b, c, k in zip(boxes, confs, kpts)]


def fuse_direction(axis_probes, direction_probes, iou_thr=0.3):
    """Axis from the OBB, direction from overlapping pose detections."""
    directed = [p for p in direction_probes if p.directed]
    if not axis_probes or not directed:
        return list(axis_probes)
    iou = box_iou([p.xyxy for p in axis_probes], [p.xyxy for p in directed])
    fused = []
    for i, probe in enumerate(axis_probes):
        hits = np.nonzero(iou[i] >= iou_thr)[0]
        if not len(hits):
            fused.append(probe)
            continue
        vote = sum(directed[j].conf * np.cos(np.radians(directed[j].angle_deg - probe.angle_deg)) for j in hits)
        angle, kpts = probe.angle_deg, probe.kpts
        if vote < 0:
            angle = float(wrap_angle(angle + 180))
            kpts = None if kpts is None else kpts[::-1].copy()
        best = directed[int(hits[iou[i, hits].argmax()])]
        fused.append(replace(probe, angle_deg=float(angle), kpts=kpts, directed=True,
                             agreement_deg=float(angular_error(angle, best.angle_deg))))
    return fused


def unflip(probe, flip_code, width, height):
    """Map a probe predicted on cv2.flip(image, flip_code) back to the original image."""
    xyxy, kpts = probe.xyxy.copy(), None if probe.kpts is None else probe.kpts.copy()
    if flip_code in (1, -1):  # horizontal
        xyxy[[0, 2]] = width - xyxy[[2, 0]]
        if kpts is not None:
            kpts[:, 0] = width - kpts[:, 0]
    if flip_code in (0, -1):  # vertical
        xyxy[[1, 3]] = height - xyxy[[3, 1]]
        if kpts is not None:
            kpts[:, 1] = height - kpts[:, 1]
    angle = probe.angle_deg if kpts is None else float(angle_from_keypoints(kpts[0], kpts[1]))
    return replace(probe, xyxy=xyxy, kpts=kpts, angle_deg=angle)
