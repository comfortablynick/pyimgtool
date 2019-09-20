"""Classes, enums, and misc data containers."""

import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)


class Position(Enum):
    """Predefined locations of text watermark."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"

    BOTTOM_LEFT = "bottom-left"

    def __str__(self):
        """Return enum value in lowercase."""
        return self.value.lower()

    def __repr__(self):
        """Return string representation of enum."""
        return str(self)

    @staticmethod
    def argparse(s):
        """Parse string values from CLI into Position."""
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
    no_op: bool = False
    pct_scale: float = 0
    width: int = 0
    height: int = 0
    keep_exif: bool = False
    watermark_text: Optional[str] = None
    watermark_image: Optional[Path] = None
    watermark_rotation: int = 0
    watermark_opacity: float = 0
    watermark_position: Optional[Position] = None
    jpg_quality: int = 0

    @staticmethod
    def from_args(args: argparse.Namespace):
        """Create Config instance from file and command line args."""
        cfg = Config()
        cfg.input_file = args.input
        cfg.output_file = args.output
        cfg.verbosity = args.verbosity
        cfg.suffix = args.suffix
        cfg.no_op = args.no_op
        cfg.pct_scale = args.pct_scale
        cfg.width = args.width
        cfg.height = args.height
        cfg.keep_exif = args.keep_exif
        cfg.watermark_text = args.watermark_text
        cfg.watermark_image = args.watermark_image
        cfg.watermark_rotation = args.watermark_rotation
        cfg.watermark_opacity = args.watermark_opacity
        cfg.watermark_position = args.watermark_position
        cfg.jpg_quality = args.jpg_quality
        return cfg


@dataclass
class ImageSize:
    """Pixel dimensions of image."""

    width: int = 0
    height: int = 0


@dataclass
class ImageContext:
    """Store details about the image being processed."""

    orig_size: ImageSize = ImageSize()
    new_size: ImageSize = ImageSize()
    orig_file_size: int = 0
    new_file_size: int = 0
    orig_dpi: Tuple[int, int] = (0, 0)
    new_dpi: Tuple[int, int] = (0, 0)
    image_buffer: Optional[bytes] = None
    orig_exif: Optional[dict] = None

    def as_dict_copy(self) -> dict:
        """Return dict representation as copy without `image_buffer` included."""
        out = self.__dict__.copy()
        try:
            del out["image_buffer"]
            del out["orig_exif"]
        except KeyError:
            pass
        return out
