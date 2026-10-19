"""Order-free keypoint loss: the two probe ends are interchangeable.

The probe's axis does not depend on which end is called the first keypoint.
With the default loss the model is punished for predicting the ends the other
way round, so when the labels disagree about which end comes first it learns
to put both points near the middle. This loss compares each prediction with
both orderings of its label and keeps the smaller error.
"""

import torch


def symmetric_keypoint_loss(sigmas):
    from ultralytics.utils.loss import KeypointLoss

    class SymmetricKeypointLoss(KeypointLoss):
        def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
            def per_object(gt, mask):
                d = (pred_kpts[..., 0] - gt[..., 0]).pow(2) + (pred_kpts[..., 1] - gt[..., 1]).pow(2)
                e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # as KeypointLoss (from cocoeval)
                return ((1 - torch.exp(-e)) * mask).sum(1) / (mask.sum(1) + 1e-9)

            # same scale as KeypointLoss: mean over objects of (sum over visible points / visible points)
            straight = per_object(gt_kpts, kpt_mask)
            swapped = per_object(gt_kpts.flip(1), kpt_mask.flip(1))
            return torch.minimum(straight, swapped).mean()

    return SymmetricKeypointLoss(sigmas)


def enable():
    """Make every pose loss created in this process order-free (for 2-keypoint models)."""
    from ultralytics.utils import loss

    if getattr(loss.v8PoseLoss, "_symmetric", False):
        return
    original_init = loss.v8PoseLoss.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.keypoint_loss = symmetric_keypoint_loss(self.keypoint_loss.sigmas)

    loss.v8PoseLoss.__init__ = __init__
    loss.v8PoseLoss._symmetric = True
