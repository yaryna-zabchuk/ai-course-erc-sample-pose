import numpy as np

from erc_pose.direction import CROP_SIZE, canonical_crop


def test_canonical_crop_puts_start_on_the_left():
    image = np.zeros((200, 200, 3), np.uint8)
    start, end = (150, 40), (50, 160)
    image[35:46, 145:156] = 255  # bright marker at the start
    crop = canonical_crop(image, start, end)
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    columns = np.nonzero(crop[..., 0].max(0) > 128)[0]
    assert columns.size and columns.max() < CROP_SIZE / 2
    flipped = canonical_crop(image, end, start)
    columns = np.nonzero(flipped[..., 0].max(0) > 128)[0]
    assert columns.min() > CROP_SIZE / 2
