"""Mat image for printing."""

import logging
import numpy as np
from typing import Tuple

from pyimgtool.data_structures import Size

LOG = logging.getLogger(__name__)


def create_mat(
    im: np.ndarray,
    size_inches: Tuple[float, float] = (8.5, 11.0),
    size_pixels: Size = None,
    dpi=300,
) -> np.ndarray:
    """Paste image onto mat of specified size.

    Parameters
    ----------
    im
        Numpy array of image data
    size_inches
        Width, height in inches (relative to `dpi`)
    size_pixels
        Size object of pixel dimensions
    dpi
        DPI of resulting image

    Returns
    -------
    np.ndarray:
        Matted image array
    """
    h, w, c = im.shape
    if size_pixels is not None:
        LOG.info("Creating mat with pixel dimensions: %s", size_pixels)
        mw, mh = tuple(size_pixels)
    else:
        LOG.info("Creating mat with print dimensions: %s", size_inches)
        mw, mh = [int(i * dpi) for i in size_inches]
    mat = 255 * np.ones((mh, mw, c), dtype=np.uint8)
    xx = (mw - w) // 2
    yy = (mh - h) // 2
    mat[yy : yy + h, xx : xx + w] = im
    return mat.copy()
