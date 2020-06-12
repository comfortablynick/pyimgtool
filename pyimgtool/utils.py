"""Helper functions."""
import logging
import platform
import re
from functools import wraps
from pathlib import PurePath

import cv2
import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger(__name__)


class Log:
    def __init__(self, logger):
        """Logging decorator.

        Logs function and parameter info.

        Parameters
        ----------
        logger
            Logger to use for logging.
        """
        self.logger = logger

    def __call__(self, fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            try:
                self.logger.debug("{0} - {1} - {2}".format(fn.__name__, args, kwargs))
                result = fn(*args, **kwargs)
                self.logger.debug(result)
                return result
            except Exception as ex:
                self.logger.debug("Exception {0}".format(ex))
                raise ex
            return result

        return decorated


def get_pkg_root() -> PurePath:
    """Return package root folder."""
    return PurePath(__file__).parent


def escape_ansi(line: str) -> str:
    """Remove ANSI escape sequences from string.

    Args:
        line: String with ansi sequences

    Returns: String with ansi sequences stripped
    """
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


def humanize_bytes(
    num: float, suffix: str = "B", si_prefix: bool = False, round_digits: int = 2
) -> str:
    """Return a human friendly byte representation.

    Modified from: https://stackoverflow.com/questions/1094841/1094933#1094933

    Args:
        num: Raw bytes
        suffix: String to append
        si_prefix: Use 1000 instead of 1024 as divisor
        round_digits: Number of decimal places to round

    Returns: Human-readable string representation of bytes
    """
    div = 1000.0 if si_prefix else 1024.0
    unit_suffix = "i" if si_prefix else ""
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < div:
            return f"{num:3.{round_digits}f} {unit}{unit_suffix}{suffix}"
        num /= div
    return f"{num:3.{round_digits}f} Y{unit_suffix}{suffix}"


def rgba2rgb(rgba: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
    """Convert RGBA image to RGB.

    Args:
        rgba: Numpy array
        background: Background color for blending

    Returns: RGB image in numpy array
    """
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def show_image_plt(im: np.ndarray):
    """Show image in matplotlib window."""
    if platform.system() != "Windows":
        LOG.info("Cannot show plot on this OS")
        return
    plt.set_loglevel("info")
    plt.figure()
    plt.imshow(im)
    plt.show()


def show_image_cv2(im: np.ndarray):
    """Show image in window."""
    if platform.system() != "Windows":
        LOG.info("Cannot show plot on this OS")
        return
    cv2.namedWindow(
        "image",
        flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
    )
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
