#!/usr/bin/env python
"""Resize and watermark images."""

import logging
import os

from PIL import Image

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

TEST_INPUT = os.path.expanduser("~/git/pyimg/test/sunset.jpg")
TEST_OUTPUT = os.path.expanduser("~/git/pyimg/test/sunset_edited.jpg")


def humanize_bytes(num, suffix="B", si_prefix=False) -> str:
    """Return a human friendly byte representation.

    Modified from: https://stackoverflow.com/questions/1094841/1094933#1094933
    """
    div = 1000.0 if si_prefix else 1024.0
    unit_suffix = "i" if si_prefix else ""
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < div:
            return f"{num:3.1f} {unit}{unit_suffix}{suffix}"
        num /= div
    return f"{num:3.1f} Y{unit_suffix}{suffix}"


def main():
    """Start point."""
    im = Image.open(TEST_INPUT)
    in_width, in_height = im.size
    LOG.info("Input dims: %s", (in_width, in_height))
    orig_size = os.path.getsize(TEST_INPUT)
    LOG.info("Input size: %s", humanize_bytes(orig_size))
    im.thumbnail((2000, 2000), Image.LANCZOS)
    im.save(TEST_OUTPUT, "JPEG")


if __name__ == "__main__":
    main()
