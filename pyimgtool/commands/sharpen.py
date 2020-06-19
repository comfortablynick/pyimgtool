"""Sharpen edges of image."""
import logging
from typing import Tuple

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


def unsharp_mask(
    im: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: float = 0.0,
) -> np.ndarray:
    """Sharpen using unsharp mask.

    Parameters
    ----------
    im
        Image data
    kernel_size
        Convolution kernel dims
    sigma
        Gaussian standard deviation
    amount
        Amount of effect to blend
    threshold
        If > 1, a low-contrast mask will be used for the effect

    Returns
    -------
    Sharpened image
    """
    LOG.info("Unsharp mask amount %f, threshold %f", amount, threshold)
    blurred = cv2.GaussianBlur(im, kernel_size, sigma)
    sharpened = float(amount + 1) * im - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    # utils.show_image_plt(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    if threshold > 0:
        low_contrast_mask = np.absolute(im - blurred)
        # blank = 255 * np.ones(im.shape, dtype=im.dtype)
        # mask = np.copyto(blank, low_contrast_mask)
        # cv2.imwrite(
        #     "../test/sharpen_mask.jpg",
        #     # np.where(low_contrast_mask < threshold)
        #     mask
        # )

        np.copyto(sharpened, im, where=low_contrast_mask < threshold)
    return sharpened
