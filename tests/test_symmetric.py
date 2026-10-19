import pytest

torch = pytest.importorskip("torch")

from ultralytics.utils.loss import KeypointLoss  # noqa: E402

from erc_pose.symmetric import symmetric_keypoint_loss  # noqa: E402


def test_symmetric_loss_ignores_keypoint_order_and_matches_default_otherwise():
    torch.manual_seed(0)
    sigmas = torch.tensor([0.5, 0.5])
    pred = torch.rand(6, 2, 3) * 10
    gt = torch.rand(6, 2, 3) * 10
    mask = torch.ones(6, 2, dtype=torch.bool)
    area = torch.rand(6, 1) * 20 + 1
    symmetric, default = symmetric_keypoint_loss(sigmas), KeypointLoss(sigmas)
    assert symmetric(pred, gt, mask, area) == pytest.approx(symmetric(pred, gt.flip(1), mask, area))
    assert symmetric(pred, gt, mask, area) <= default(pred, gt, mask, area) + 1e-6
    exact = pred.clone()
    assert symmetric(exact, exact.flip(1), mask, area) == pytest.approx(0, abs=1e-6)
