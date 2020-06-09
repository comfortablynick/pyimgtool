"""Configuration and argument parsing."""

import argparse
import logging
import sys
import textwrap
from shutil import get_terminal_size
from typing import List, Tuple

from pyimgtool.data_structures import Position
from pyimgtool.version import __version__

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.RawTextHelpFormatter):
    """Format help messages to custom spec."""

    def __init__(self, prog):
        """Override formatter defaults."""
        max_width = min(100, get_terminal_size()[0])
        # TODO: get all help items and set max_width based on longest (or term width)
        # max_help_position >= max(len(param.name)+len(param.metavar) for param in params)
        super(CustomFormatter, self).__init__(
            prog, max_help_position=max_width, width=max_width
        )
        self._max_help_position = max_width
        self._action_max_length += 4

    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(option_string)
                parts[-1] += f" {args_string}"
            return ", ".join(parts)

    def _get_help_string(self, action):
        help = action.help
        if "%(default)" not in action.help:
            if action.default not in [argparse.SUPPRESS, None, False]:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += " (default: %(default)s)"
        return help


class OrderedNamespace(argparse.Namespace):
    """Namespace that retains calling order of subparsers.

    Args:
        commands: Subparsers containing commands to parse
    """

    def __init__(self, commands, **kwargs):
        """Pass possible commands to namespace."""
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
        # TODO: find way to put top-level attrs in own namespace
        return ((attr, getattr(self, attr)) for attr in ["_top_level"] + self._order)


def split_to_tuple(arg: str) -> Tuple[float, ...]:
    """Split string `arg` into tuple using `delimiter`.

    Parameters
    ----------
    arg
        Command line argument.

    Returns
    -------
    Tuple of split args
    """
    return tuple([float(i) for i in arg.split(",")])


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
        description=desc, formatter_class=argparse.RawTextHelpFormatter
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
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(
        title="Commands", description="image operations", help="valid commands",
    )

    # Commands
    # Open
    open_cmd = commands.add_parser("open", help="open image for editing")
    open_cmd.add_argument(
        "input",
        help="image file to process",
        type=argparse.FileType(mode="rb"),
        metavar="INPUT_FILE",
    )
    open_cmd.add_argument(
        "-H",
        "--histogram",
        help="print image histogram to console",
        dest="show_histogram",
        action="store_true",
    )

    # Mat
    mat_cmd = commands.add_parser(
        "mat", help="add a mat of a specific size for printing"
    )
    mat_cmd.add_argument(
        "size",
        help="dimensions of mat, in inches",
        metavar="SIZE",
        type=split_to_tuple,
    )
    mat_cmd.add_argument(
        "-d", "--dpi", help="Dots per inch for mat", type=int, default=300
    )
    # Resize
    resize_cmd = commands.add_parser("resize", help="resize image dimensions",)
    resize_cmd.add_argument(
        "-s",
        "--scale",
        help="scale output size",
        dest="scale",
        metavar="SCALE",
        type=float,
    )
    resize_cmd.add_argument(
        "-W", "--width", help="absolute width of output", metavar="PX", type=int,
    )
    resize_cmd.add_argument(
        "-H", "--height", help="absolute height of output", metavar="PX", type=int,
    )
    resize_cmd.add_argument(
        "-L", "--longest", help="longest dimension of output", metavar="PX", type=int,
    )
    resize_cmd.add_argument(
        "-S", "--shortest", help="shortest dimension of output", metavar="PX", type=int,
    )

    # Resize2
    resize2_cmd = commands.add_parser(
        "resize2", help="resize image dimensions using opencv",
    )
    resize2_cmd.add_argument(
        "-s", "--scale", help="scale output size", metavar="SCALE", type=float,
    )
    resize2_cmd.add_argument(
        "-W", "--width", help="absolute width of output", metavar="PX", type=int,
    )
    resize2_cmd.add_argument(
        "-H", "--height", help="absolute height of output", metavar="PX", type=int,
    )
    resize2_cmd.add_argument(
        "-L", "--longest", help="longest dimension of output", metavar="PX", type=int,
    )
    resize2_cmd.add_argument(
        "-S", "--shortest", help="shortest dimension of output", metavar="PX", type=int,
    )

    # Watermark
    watermark_cmd = commands.add_parser("watermark", help="add watermark to image")
    watermark_cmd.add_argument(
        "image",
        help="image file to use as watermark",
        type=argparse.FileType("rb"),
        metavar="IMAGE",
    )
    watermark_cmd.add_argument(
        "-r",
        "--rotation",
        help="angle of watermark rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    watermark_cmd.add_argument(
        "-o", "--opacity", help="watermark opacity", type=float, default=0.3,
    )
    watermark_cmd.add_argument(
        "-p",
        "--position",
        help="watermark position",
        metavar="POSITION",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    watermark_cmd.add_argument(
        "-s",
        "--scale",
        help="watermark scale in percent of image size",
        default=0.2,
        type=float,
    )

    # Watermark2
    watermark2_cmd = commands.add_parser(
        "watermark2", help="add watermark to image using numpy and opencv"
    )
    watermark2_cmd.add_argument(
        "image",
        help="image file to use as watermark",
        type=argparse.FileType("rb"),
        metavar="IMAGE",
    )
    watermark2_cmd.add_argument(
        "-r",
        "--rotation",
        help="angle of watermark rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    watermark2_cmd.add_argument(
        "-o", "--opacity", help="watermark opacity", type=float, default=0.3,
    )
    watermark2_cmd.add_argument(
        "-p",
        "--position",
        help="watermark position",
        metavar="POSITION",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    watermark2_cmd.add_argument(
        "-s",
        "--scale",
        help="watermark scale in percent of image size",
        dest="scale",
        metavar="SCALE",
        default=0.2,
        type=float,
    )

    # Text
    text_cmd = commands.add_parser("text", help="add text to image")
    text_cmd.add_argument(
        "text", help="text to display on image", metavar="TEXT", type=str
    )
    text_cmd.add_argument(
        "-c",
        "--copyright",
        help="display TEXT as copyright message after © and date taken",
        action="store_true",
    )
    text_cmd.add_argument(
        "-r",
        "--rotation",
        help="angle of text rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    text_cmd.add_argument(
        "-o",
        "--opacity",
        help="opacity of text layer",
        type=float,
        metavar="OPACITY",
        default=0.3,
    )
    text_cmd.add_argument(
        "-p",
        "--position",
        help="position of text",
        metavar="POSITION",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    text_cmd.add_argument(
        "-s",
        "--scale",
        help="scale of text relative to image width",
        default=0.20,
        type=float,
    )

    # Text2
    text2_cmd = commands.add_parser("text2", help="add text to image using opencv")
    text2_cmd.add_argument(
        "text", help="text to display on image", metavar="TEXT", type=str
    )
    text2_cmd.add_argument(
        "-c",
        "--copyright",
        help="display TEXT as copyright message after © and date taken",
        dest="copyright",
        action="store_true",
    )
    text2_cmd.add_argument(
        "-r",
        "--rotation",
        help="angle of text rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    text2_cmd.add_argument(
        "-o", "--opacity", help="opacity of text layer", type=float, default=0.3,
    )
    text2_cmd.add_argument(
        "-p",
        "--position",
        help="position of text",
        metavar="POSITION",
        default=Position.BOTTOM_RIGHT,
        type=Position.argparse,
        choices=list(Position),
    )
    text2_cmd.add_argument(
        "-s",
        "--scale",
        help="scale of text relative to image width",
        default=0.20,
        type=float,
    )

    # Save
    save_cmd = commands.add_parser("save", help="save edited file to disk")
    save_cmd.add_argument(
        "output",
        help="file to save processed image",
        type=argparse.FileType(mode="w"),
        metavar="OUTPUT_FILE",
    )
    save_cmd.add_argument(
        "-f",
        "--force",
        help="force overwrite of existing file",
        dest="force",
        action="store_true",
    )
    save_cmd.add_argument(
        "-k",
        "--keep-exif",
        help="keep exif data if possible",
        dest="keep_exif",
        action="store_true",
    )
    save_cmd.add_argument(
        "-n",
        "--noop",
        help="display results only; don't save file",
        dest="no_op",
        action="store_true",
    )
    save_cmd.add_argument(
        "-q",
        help="Quality setting for jpeg files (an integer between 1 and 100; default: 75)",
        type=lambda n: 0 <= int(n) <= 100
        or parser.error("quality must be between 0 and 100"),
        dest="jpg_quality",
        default=75,
        metavar="QUALITY",
    )
    save_cmd.add_argument(
        "-s",
        nargs=1,
        help="text suffix appended to INPUT path if no OUTPUT file given",
        metavar="TEXT",
        dest="suffix",
        default="_edited",
    )

    for _, subp in commands.choices.items():
        subp.formatter_class = CustomFormatter

    if parser._positionals.title is not None:
        parser._positionals.title = "Arguments"
    if parser._optionals.title is not None:
        parser._optionals.title = "Options"

    if not args:
        print("error: arguments required", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        sys.exit(1)

    # user-defined opts on the main parser that we will remove from
    # the namespaces of the lower-level parsers (not sure why they appear)
    top_level_opts = [
        action.dest
        for action in parser._actions
        if action.dest not in ["help", "version", "==SUPPRESS=="]
    ]

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
        for opt in top_level_opts:
            delattr(n, opt)
    # setattr(ns, '_top', )
    # basic validation
    if ns.quiet > 0:
        ns.verbosity = 0

    # give top-level args their own namespace
    top = argparse.Namespace()
    for k, v in ns.__dict__.items():
        if k in top_level_opts:
            setattr(top, k, v)
            # delattr(ns, k)
    setattr(ns, "_top_level", top)
    for a in top_level_opts:
        delattr(ns, a)
    return ns


# vim:fdl=3:
