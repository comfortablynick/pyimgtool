"""Classes, enums, and misc data containers."""

import logging
from argparse import Namespace as ArgparseNamespace
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

LOG = logging.getLogger(__name__)


class Position(Enum):
    """Predefined locations of text watermark."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"

    def __str__(self):
        """Return enum value in lowercase."""
        return self.value.lower()

    def __repr__(self):
        """Return string representation of enum."""
        return str(self)

    @staticmethod
    def argparse(s):
        """Parse string values from CLI into Position.

        Args:
            s: Value to match against enum attribute
        """
        vals = {x.value.lower(): x for x in list(Position)}
        try:
            return vals[s.lower()]
        except KeyError:
            return s


@dataclass
class Config:
    """Store options from config file and command line."""

    input_file: Optional[Path] = None
    output_file: Optional[Path] = None
    verbosity: int = 0
    suffix: Optional[str] = None
    force: bool = False
    no_op: bool = False
    jpg_quality: int = 0
    commands: Optional[List[str]] = None
    pct_scale: float = 0
    width: int = 0
    height: int = 0
    keep_exif: bool = False
    show_histogram: bool = False
    watermark_image: Optional[Path] = None
    watermark_rotation: int = 0
    watermark_opacity: float = 0
    watermark_position: Optional[Position] = None
    watermark_scale: float = 0
    watermark_padding: int = 10
    text: Optional[str] = None
    text_copyright: Optional[str] = None
    text_rotation: int = 0
    text_opacity: float = 0
    text_position: Optional[Position] = None
    text_scale: int = 0
    text_padding: int = 0

    @staticmethod
    def from_args(args: ArgparseNamespace):
        """Create Config instance from file and command line args."""
        cfg = Config()
        cfg.input_file = args.input
        cfg.output_file = args.output
        cfg.verbosity = args.verbosity
        cfg.suffix = args.suffix
        cfg.force = args.force
        cfg.no_op = args.no_op
        cfg.jpg_quality = args.jpg_quality
        cfg.commands = args.command
        cfg.show_histogram = args.show_histogram
        if args.command is not None:
            if "resize" in args.command:
                cfg.pct_scale = args.pct_scale
                cfg.width = args.width
                cfg.height = args.height
                cfg.keep_exif = args.keep_exif
            if "watermark" in args.command:
                cfg.watermark_image = args.watermark_image
                cfg.watermark_rotation = args.watermark_rotation
                cfg.watermark_opacity = args.watermark_opacity
                cfg.watermark_position = args.watermark_position
                cfg.watermark_scale = args.watermark_scale
            if "text" in args.command:
                cfg.text = args.text
                cfg.text_copyright = args.text_copyright
                cfg.text_rotation = args.text_rotation
                cfg.text_opacity = args.text_opacity
                cfg.text_position = args.text_position
                cfg.text_scale = args.text_scale
        return cfg


@dataclass
class ImageSize:
    """Pixel dimensions of image."""

    width: int = 0
    height: int = 0

    def __str__(self):
        """Return string representation, e.g.: width x height px."""
        return f"{self.width} x {self.height} px"

    @property
    def area(self) -> int:
        """Pixel area of image."""
        return self.width * self.height


@dataclass
class Context:
    """Keep track of details of processing for later reporting."""

    orig_size: ImageSize = field(default_factory=ImageSize)
    new_size: ImageSize = field(default_factory=ImageSize)
    watermark_size: ImageSize = field(default_factory=ImageSize)
    orig_file_size: int = 0
    new_file_size: int = 0
    orig_dpi: Tuple[int, int] = (0, 0)
    new_dpi: Tuple[int, int] = (0, 0)
    image_buffer: Optional[bytes] = field(default=None, repr=False)
    orig_exif: Optional[dict] = field(default=None, repr=False)
    time_start: float = 0
    time_end: float = 0
