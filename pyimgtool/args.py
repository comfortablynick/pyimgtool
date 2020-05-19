"""Configuration and argument parsing."""

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import List

from pyimgtool.data_structures import Position
from pyimgtool.version import __version__

LOG = logging.getLogger(__name__)


def parse_args(args: list) -> List[argparse.Namespace]:
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

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        description="image operations",
        help="valid commands",
        # metavar="<command>"
    )

    # input
    open = subparsers.add_parser("open", help="open image for editing")
    open.add_argument(
        "input",
        help="image file to process",
        type=argparse.FileType(mode="rb"),
        nargs=1,
        metavar="INPUT_FILE",
    )
    open.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    open.add_argument(
        "-v",
        help="increase logging output to console",
        action="count",
        dest="verbosity",
        default=0,
    )
    # parser.add_argument(
    #     "-n",
    #     "--noop",
    #     help="display results only; don't save file",
    #     dest="no_op",
    #     action="store_true",
    # )
    open.add_argument(
        "-Q",
        "--quiet",
        help="quiet debug log output to console (opposite of -v)",
        action="store_true",
        dest="quiet",
    )
    open.add_argument(
        "-H",
        "--histogram",
        help="print image histogram to console",
        dest="show_histogram",
        action="store_true",
    )

    # resize
    resize = subparsers.add_parser("resize", help="resize image dimensions")
    resize.add_argument(
        "-s",
        help="scale output size",
        dest="scale",
        metavar="SCALE",
        type=lambda n: 0 < float(n) < 1 or parser.error("resize scale must be between 0 and 1"),
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

    # Watermark
    watermark = subparsers.add_parser("watermark", help="add watermark to image")
    watermark.add_argument(
        "-i",
        help="image file to use as watermark",
        type=Path,
        dest="watermark_image",
        metavar="PATH",
    )
    watermark.add_argument(
        "-r",
        help="angle of watermark rotation",
        dest="watermark_rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    watermark.add_argument(
        "-o",
        help="watermark opacity",
        dest="watermark_opacity",
        type=float,
        metavar="OPACITY",
        default=0.3,
    )
    watermark.add_argument(
        "-p",
        help="watermark position",
        dest="watermark_position",
        metavar="POS",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    watermark.add_argument(
        "-s",
        help="watermark scale in percent of image size (default = 10)",
        dest="watermark_scale",
        metavar="SCALE",
        default=0.2,
        type=lambda n: 0 < float(n) < 1 or parser.error("watermark scale must be between 0 and 1"),
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
        type=lambda n: 0 < float(n) < 1 or parser.error("text scale must be between 0 and 1"),
    )

    save = subparsers.add_parser("save", help="save edited file to disk")
    save.add_argument(
        "output",
        help="file to save processed image",
        type=argparse.FileType(mode="w"),
        metavar="OUTPUT_FILE",
    )
    save.add_argument(
        "-f",
        "--force",
        help="force overwrite of existing file",
        dest="force",
        action="store_true",
    )
    save.add_argument(
        "-k", help="keep exif data if possible", dest="keep_exif", action="store_true"
    )
    save.add_argument(
        "-q",
        help="Quality setting for jpeg files (an integer between 1 and 100; default: 75)",
        type=lambda n: 0 <= int(n) <= 100
        or parser.error("quality must be between 0 and 100"),
        dest="jpg_quality",
        default=75,
        metavar="QUALITY",
    )
    save.add_argument(
        "-s",
        nargs=1,
        help="text suffix appended to INPUT path if no OUTPUT file given",
        metavar="TEXT",
        dest="suffix",
        default="_edited",
    )

    if parser._positionals.title is not None:
        parser._positionals.title = "Arguments"
    if parser._optionals.title is not None:
        parser._optionals.title = "Options"
    namespaces = []
    if not args:
        print("error: arguments required", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        sys.exit(1)
    while args:
        parsed, args = parser.parse_known_args(args)
        namespaces.append(parsed)

    # do basic validation
    if namespaces[0].command != "open":
        print("error: open command must be called first", file=sys.stderr)
        sys.exit(1)
    if namespaces[0].quiet > 0:
        namespaces[0].verbosity = 0

    # if not 0 <= parsed.jpg_quality <= 100:
    #     parser.error(f"Quality (-q) must be within 0-100; found: {parsed.jpg_quality}")

    # if parsed.pct_scale and (parsed.width or parsed.height):
    #     parser.error("Can use either -p or -mw/-mh, not both")
    # if not 0 <= parsed.watermark_scale <= 1:
    #     parser.error("Value out of bounds: -ws must be between 0 and 1")
    #
    # if parsed.text is not None and parsed.text_copyright is not None:
    #     parser.error("Can use either -t or -c, not both")
    return namespaces
