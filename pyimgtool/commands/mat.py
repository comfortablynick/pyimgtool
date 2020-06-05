"""Mat image for printing."""

import logging
import numpy as np

LOG = logging.getLogger(__name__)


def create_mat(im: np.ndarray, mat_size: str = "letter", dpi=300, portrait=False) -> np.ndarray:
    """Paste image onto mat of a specified size.

    Args:
        im: Numpy array of image data
        mat_size: Preset sizes of mat
        dpi: DPI of resulting image

    Returns: matted image array
    """
    h, w, c = im.shape
    if mat_size == "letter":
        mh = int(round(dpi * 8.5))
        mw = int(round(dpi * 11.0))
    else:
        mh = 5000
        mw = 5000
    if portrait:
        mw, mh = mh, mw
    mat = 255 * np.ones((mh, mw, c), dtype=np.uint8)
    xx = (mw - w) // 2
    yy = (mh - h) // 2
    mat[yy : yy + h, xx : xx + w] = im
    return mat.copy()
