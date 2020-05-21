"""Configuration and argument parsing."""

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

from pyimgtool.data_structures import Position
from pyimgtool.version import __version__

LOG = logging.getLogger(__name__)


class OrderedNamespace(argparse.Namespace):
    """Namespace that retains calling order of subparsers."""

    def __init__(self, commands, **kwargs):
        """Pass possible commands to namespace.
        
        Args:
            commands: Subparsers containing commands to parse
        """
        self.__dict__["_order"] = []
        self._commands = [*commands.choices]
        super().__init__(**kwargs)

    def __setattr__(self, attr, value):
        """Set order of commands called."""
        super().__setattr__(attr, value)
        if attr in self._commands:
            if attr in self._order:
                self.__dict__["_order"].clear()
            self.__dict__["_order"].append(attr)

    def ordered(self):
        """Return namespace in order.

        Yield: Command namespace in the order called
        """
        return ((attr, getattr(self, attr)) for attr in self._order)


def parse_args(args: List[str]) -> OrderedNamespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments

    Return: 2-tuple of Argparse namespace of parsed arguments and list of commands called
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
    parser.add_argument(
        "-v",
        help="increase logging output to console",
        action="count",
        dest="verbosity",
        default=0,
    )
    parser.add_argument(
        "-Q",
        "--quiet",
        help="quiet debug log output to console (opposite of -v)",
        action="store_true",
        dest="quiet",
    )
    commands = parser.add_subparsers(
        title="Commands", description="image operations", help="valid commands",
    )

    # input
    open = commands.add_parser("open", help="open image for editing")
    open.add_argument(
        "input",
        help="image file to process",
        type=argparse.FileType(mode="rb"),
        metavar="INPUT_FILE",
    )
    open.add_argument(
        "-H",
        "--histogram",
        help="print image histogram to console",
        dest="show_histogram",
        action="store_true",
    )
    open.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # resize
    resize = commands.add_parser("resize", help="resize image dimensions")
    resize.add_argument(
        "-s", help="scale output size", dest="scale", metavar="SCALE", type=float,
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
    watermark = commands.add_parser("watermark", help="add watermark to image")
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
        type=lambda n: 0 < float(n) < 1
        or parser.error("watermark scale must be between 0 and 1"),
    )

    # Text
    text = commands.add_parser("text", help="add text to image")
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
        type=lambda n: 0 < float(n) < 1
        or parser.error("text scale must be between 0 and 1"),
    )

    save = commands.add_parser("save", help="save edited file to disk")
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
        "-k",
        "--keep-exif",
        help="keep exif data if possible",
        dest="keep_exif",
        action="store_true",
    )
    save.add_argument(
        "-n",
        "--noop",
        help="display results only; don't save file",
        dest="no_op",
        action="store_true",
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

    if not args:
        print("error: arguments required", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        sys.exit(1)

    # opts = parser._optionals
    # split argv by known commands and parse
    split_argv: List[List] = [[]]
    commands_found = []
    for c in sys.argv[1:]:
        if c in commands.choices:
            commands_found.append(c)
            if c == "-h" and len(split_argv) >= 1:
                split_argv[-1].append(c)
            else:
                split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    ns = OrderedNamespace(commands)
    for c in commands.choices:
        setattr(ns, c, None)
    # Parse each command
    if len(split_argv) == 0:
        split_argv.append(["-h"])  # if no command was given
    parser.parse_args(split_argv[0], namespace=ns)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(ns, argv[0], n)
        parser.parse_args(argv, namespace=n)
    # basic validation
    print(ns)
    if ns.quiet > 0:
        ns.verbosity = 0
    return ns
