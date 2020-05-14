"""Configuration and argument parsing."""

import argparse
import logging
import textwrap
from pathlib import Path

from pyimgtool.data_structures import Position
from pyimgtool.version import __version__

LOG = logging.getLogger(__name__)


def parse_args(args: list) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments

    Return: Argparse namespace of parsed arguments
    """
    # flags
    desc = textwrap.dedent(
        """\
        A command-line utility which uses the Pillow module to
        manipulate images for web vewing.

        Images can be resampled, resized, and compressed at custom
        quality levels. Watermarking can also be added.
        """
    )
    parser = argparse.ArgumentParser(
        prog="pyimgtool",
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Optional flags
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-c",
        help="read from this config file (default: conf.ini)",
        dest="config_file",
        default="../conf.ini",
    )
    parser.add_argument(
        "-f", help="force overwrite of existing file", dest="force", action="store_true"
    )
    parser.add_argument(
        "-v",
        help="increase logging output to console",
        action="count",
        dest="verbosity",
        default=0,
    )
    parser.add_argument(
        "-s",
        nargs=1,
        help="text suffix appended to INPUT path if no OUTPUT file given",
        metavar="TEXT",
        dest="suffix",
        default="_edited",
    )
    parser.add_argument(
        "-n",
        help="display results only; don't save file",
        dest="no_op",
        action="store_true",
    )
    parser.add_argument(
        "-Q",
        help="quiet debug log output to console (opposite of -v)",
        action="store_true",
        dest="quiet",
    )
    parser.add_argument(
        "-H",
        help="print image histogram to console",
        dest="show_histogram",
        action="store_true",
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        description="image operations",
        help="valid commands",
        # metavar="<command>"
    )

    common = parser.add_argument_group("Common parameters")
    common.add_argument("-i", help="image file to process", action="store", dest="input", metavar="INPUT")
    common.add_argument("-o", help="file to save processed image", action="store", dest="output", metavar="OUTPUT")
    common.add_argument(
        "-q",
        help="Quality setting for jpeg files (an integer between 1 and 100; default: 75)",
        type=int,
        dest="jpg_quality",
        default=75,
        metavar="QUALITY",
    )


    # resize
    resize = subparsers.add_parser("resize", help="resize image dimensions")
    resize.add_argument(
        "-p",
        help="scale output by percent of orig size",
        dest="pct_scale",
        metavar="SCALE",
        type=float,
    )
    resize.add_argument(
        "-mw",
        help="maximum width of output",
        dest="width",
        metavar="WIDTH",
        type=int,
        default=0,
    )
    resize.add_argument(
        "-mh",
        help="maximum height of output",
        dest="height",
        metavar="HEIGHT",
        type=int,
        default=0,
    )
    resize.add_argument(
        "-ld",
        help="longest dimension of output",
        dest="longest_dim",
        metavar="PIXELS",
        type=int,
        default=0,
    )
    resize.add_argument(
        "-ke", help="keep exif data if possible", dest="keep_exif", action="store_true"
    )

    # Watermark
    watermark = subparsers.add_parser("watermark", help="add watermark to image")
    watermark.add_argument(
        "-wi",
        help="image file to use as watermark",
        type=Path,
        dest="watermark_image",
        metavar="PATH",
    )
    watermark.add_argument(
        "-wr",
        help="angle of watermark rotation",
        dest="watermark_rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    watermark.add_argument(
        "-wo",
        help="watermark opacity",
        dest="watermark_opacity",
        type=float,
        metavar="OPACITY",
        default=0.3,
    )
    watermark.add_argument(
        "-wp",
        help="watermark position",
        dest="watermark_position",
        metavar="POS",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    watermark.add_argument(
        "-ws",
        help="watermark scale in percent of image size (default = 10)",
        dest="watermark_scale",
        metavar="SCALE",
        default=0.2,
        type=float,
    )

    # Text
    text = subparsers.add_parser("text", help="add text to image")
    text.add_argument(
        "-t", help="text to display on image", dest="text", metavar="TEXT", type=str
    )
    text.add_argument(
        "-c",
        help="display copyright message after © and date taken",
        dest="text_copyright",
        metavar="MESSAGE",
        type=str,
    )
    text.add_argument(
        "-r",
        help="angle of text rotation",
        dest="text_rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    text.add_argument(
        "-o",
        help="text opacity",
        dest="text_opacity",
        type=float,
        metavar="OPACITY",
        default=0.3,
    )
    text.add_argument(
        "-p",
        help="text position",
        dest="text_position",
        metavar="POS",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    text.add_argument(
        "-s",
        help="scale of text relative to image width",
        dest="text_scale",
        metavar="SCALE",
        default=0.20,
        type=float,
    )

    if parser._positionals.title is not None:
        parser._positionals.title = "Arguments"
    if parser._optionals.title is not None:
        parser._optionals.title = "Options"
    parsed = parser.parse_args(args)

    # do basic validation
    if parsed.quiet > 0:
        parsed.verbosity = 0

    if not 0 <= parsed.jpg_quality <= 100:
        parser.error(f"Quality (-q) must be within 0-100; found: {parsed.jpg_quality}")

    if parsed.pct_scale and (parsed.width or parsed.height):
        parser.error("Can use either -p or -mw/-mh, not both")
    if not 0 <= parsed.watermark_scale <= 1:
        parser.error("Value out of bounds: -ws must be between 0 and 1")

    if parsed.text is not None and parsed.text_copyright is not None:
        parser.error("Can use either -t or -c, not both")

    return parsed
