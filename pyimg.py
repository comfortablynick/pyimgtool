#!/usr/bin/env python
"""Resize and watermark images."""

import argparse
import logging
import os
import sys
from enum import Enum

from PIL import Image

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class Position(Enum):
    """Predefined locations of text watermark."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"


def parse_args(args: list):
    """Parse command line arguments."""
    # flags
    desc = (
        "A command-line utility which uses the vips library to manipulate "
        "images for web vewing. Images can be resampled, resized, and "
        "compressed at custom quality levels. Watermarking can also be added."
    )
    parser = argparse.ArgumentParser(prog="pyimg", description=desc, add_help=False)

    # Positionals
    parser.add_argument("input", help="image file to process", metavar="INPUT")
    parser.add_argument("output", help="file to save processed image", metavar="OUTPUT")

    # Flags
    parser.add_argument(
        "-v",
        help="increase logging output to console",
        action="count",
        dest="verbosity",
        default=0,
    )
    parser.add_argument("-h", action="help", help="show this help message and exit")
    parser.add_argument("--help", action="help", help=argparse.SUPPRESS)
    parser.add_argument("-V", action="version", version="%(prog)s 0.0.1")
    parser.add_argument(
        "-s",
        nargs=1,
        help="text suffix appended to INPUT path if no OUTPUT file given",
        metavar="TEXT",
        default="_edited",
    )
    parser.add_argument(
        "-n",
        help="display results only; don't save file",
        dest="no_op",
        action="store_true",
    )

    # Image group
    image_group = parser.add_argument_group("General image options")
    image_group.add_argument(
        "-ke", help="keep exif data", dest="keep_exif", action="store_true"
    )
    image_group.add_argument(
        "-mw",
        help="maximum width of output",
        dest="width",
        metavar="WIDTH",
        type=int,
        default=0,
    )
    image_group.add_argument(
        "-mh",
        help="maximum height of output",
        dest="height",
        metavar="HEIGHT",
        default=0,
        type=int,
    )
    image_group.add_argument(
        "-wt",
        help="text to display in watermark",
        type=str,
        dest="watermark_text",
        metavar="TEXT",
    )
    image_group.add_argument(
        "-wr",
        help="angle of watermark rotation",
        dest="watermark_rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    image_group.add_argument(
        "-wp",
        help="watermark position",
        dest="watermark_position",
        metavar="POS",
        default=Position.BOTTOM_RIGHT,
    )

    # Jpg group
    jpg_group = parser.add_argument_group("Jpeg options")
    jpg_group.add_argument(
        "-q",
        help="Quality setting for jpeg files (an integer between 1 and 100; default: 75)",
        type=int,
        dest="jpg_quality",
        default=75,
        metavar="QUALITY",
    )

    if parser._positionals.title is not None:
        parser._positionals.title = "Arguments"
    if parser._optionals.title is not None:
        parser._optionals.title = "Options"
    parsed = parser.parse_intermixed_args(args)

    # do rudimentary checks
    if not 0 <= parsed.jpg_quality <= 100:
        parser.exit(
            1, f"Quality (-q) must be within 0-100; found: {parsed.jpg_quality}\n"
        )
    return parsed


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
    args = parse_args(sys.argv[1:])
    im = Image.open(args.input)
    in_width, in_height = im.size
    LOG.info("Input dims: %s", (in_width, in_height))
    orig_size = os.path.getsize(args.input)
    LOG.info("Input size: %s", humanize_bytes(orig_size))
    im.thumbnail((2000, 2000), Image.LANCZOS)
    im.save(args.output, "JPEG")


if __name__ == "__main__":
    main()
