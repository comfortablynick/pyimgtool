"""Helper functions."""
import re
import numpy as np
from pathlib import PurePath


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
