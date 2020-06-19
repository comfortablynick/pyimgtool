"""Helper functions."""
import logging
import platform
import re
import time
from functools import wraps
from pathlib import PurePath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotille

LOG = logging.getLogger(__name__)


class Log:
    """Decorator class to log fn execution time and parameters."""

    def __init__(self, logger=None, fmt="[{elapsed:0.8f}s] {name}({args}) -> {result}"):
        self.fmt = fmt
        self.logger = logger
        np.set_string_function(np_repr)

    def __call__(self, func):
        if self.logger is None:
            logging.basicConfig(level=10)
            self.logger = logging.getLogger(__name__)

        @wraps(func)
        def wrapper(*_args, **_kwargs):
            t0 = time.time()
            _result = func(*_args, **_kwargs)
            elapsed = time.time() - t0
            name = func.__name__
            arg_lst = []
            if _args is not None:
                arg_lst.append(", ".join(repr(a) for a in _args))
            if _kwargs is not None:
                arg_lst.append(
                    ", ".join(f"{k}={repr(w)}" for k, w in sorted(_kwargs.items()))
                )
            args = ", ".join(arg_lst)
            result = repr(_result)
            self.logger.debug(self.fmt.format(**locals()))
            return _result

        return wrapper


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
    col, row, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((col, row, 3), dtype=np.float64)
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype=np.float64) / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype=np.uint8)


def bgr2gray(im: np.ndarray, weights=[0.1140, 0.5870, 0.2989]) -> np.ndarray:
    """Transform BGR image into grayscale."""
    tile = np.tile(weights, reps=(im.shape[0], im.shape[1], 1))
    return np.sum(tile * im, axis=2)


def equalize_hsv(image):
    """Equalize histogram of image."""
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image


def show_image_plt(im: np.ndarray):
    """Show image in matplotlib window."""
    if platform.system() != "Windows":
        LOG.info("Cannot show plot on this OS")
        return
    plt.set_loglevel("info")
    plt.figure()
    cmap = None
    if len(im.shape) == 2:
        cmap = "gray"
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im.astype(np.uint8), cmap=cmap, vmin=0, vmax=255)
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
    cv2.imshow("image", im.astype(np.uint8))

    cv2.waitKey(0)


def show_histogram(im: np.ndarray):
    """Show gui histogram plot.

    Parameters
    ----------
    im : np.ndarray
        Image
    """
    if platform.system() != "Windows":
        return
    if len(im.shape) > 2:
        LOG.debug("Converting image to grayscale for histogram plot")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    assert len(im.shape) == 2, "Only grayscale images can be used with this func"
    hist, bins = np.histogram(im, bins=256, range=(0, 255))
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align="center")
    plt.title("Histogram (Luminosity)")
    plt.xlabel("Pixel Value")
    plt.ylabel("# of Pixels")
    plt.show()


def show_position_histograms(im: np.ndarray, positions):
    """Show gui histogram plot for multiple positions in image.

    Parameters
    ----------
    im : np.ndarray
        Image
    """
    if platform.system() != "Windows":
        return
    if len(im.shape) > 2:
        LOG.debug("Converting image to grayscale for histogram plot")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    assert len(im.shape) == 2, "Only grayscale images can be used with this func"
    fig, axs = plt.subplots(3, 2)
    plt.title("Histogram (Luminosity)")
    for ax in axs.flat:
        ax.set(xlabel="Pixel Value", ylabel="# of Pixels")

    for pos in positions:
        hist, bins = np.histogram(im, bins=256, range=(0, 255))
        center = (bins[:-1] + bins[1:]) / 2
        # print(bins)
        plt.bar(center, hist, align="center")
    plt.show()


def show_rgb_histogram(im: np.ndarray):
    """Display rgb histogram plot.

    Parameters
    ----------
    im : np.ndarray
        Image to plot
    """
    if platform.system() != "Windows":
        return
    chans = cv2.split(im)
    colors = ("r", "g", "b")
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("# of Pixels")
    plt.xlim([0, 256])
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        # hist, bins = np.histogram(chan, bins=256, range=(0, 256))
        # plot the histogram
        plt.plot(hist, color=color)
    plt.show()


def np_repr(array: np.ndarray) -> str:
    """Custom repr for numpy.ndarray.

    The string is formatted to show "ndarray((h, w[, c]), dtype=dtype)"

    Parameters
    ----------
    array
        Numpy ndarray

    Returns
    -------
    Formatted string
    """
    return f"ndarray({array.shape} dtype={array.dtype})"


def generate_rgb_histogram(im: np.ndarray, show_axes: bool = False) -> str:
    """Generate histogram for terminal.

    Parameters
    ----------
    im
        Image to evaluate
    show_axes
        Show x and y axis labels

    Returns
    -------
    str:
        Histogram to print
    """
    img = np.asarray(im)
    hist_width = 50
    hist_height = 10
    hist_bins = 256

    # set up graph
    fig = plotille.Figure()
    fig.width = hist_width
    fig.height = hist_height
    fig.origin = False  # Don't draw 0 lines
    fig.set_x_limits(min_=0, max_=hist_bins - 1)
    fig.set_y_limits(min_=0)
    fig.color_mode = "names"

    img_h, img_w, img_c = img.shape
    colors = ["red", "green", "blue"]
    chans = cv2.split(img)

    # for i in range(img_c):
    for (chan, color) in zip(chans, colors):
        hist_data, bins = np.histogram(chan, bins=hist_bins, range=(0, 255))
        fig.plot(bins[:hist_bins], hist_data, lc=color)
    if not show_axes:
        graph = (
            "\n".join(
                ["".join(ln.split("|")[1]) for ln in fig.show().splitlines()[1:-2]]
            )
            + "\n"
        )

    else:
        graph = fig.show()
    return graph
