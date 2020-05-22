"""Classes, enums, and misc data containers."""

import logging
from argparse import Namespace as ArgparseNamespace
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

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
