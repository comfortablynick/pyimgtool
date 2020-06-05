"""Classes, enums, and misc data containers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

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
class Size:
    """Pixel dimensions of image.

    Args:
        width: Width of image (None -> 0)
        height: Height of image (None -> 0)
    """

    def __init__(self, width: int = None, height: int = None):
        self.width = width if width is not None else 0
        self.height = height if height is not None else 0

    def __iter__(self):
        """Allow iteration of object."""
        for item in [self.width, self.height]:
            yield item

    def __getitem__(self, item):
        """Access class items by key or index."""
        try:
            return getattr(self, item)
        except TypeError:
            return tuple(self)[item]

    def __str__(self):
        """Return string representation, e.g.: width x height px."""
        return f"{self.width} x {self.height} px"

    def __lt__(self, other):
        return self.height < other.height or self.width < other.width

    def __gt__(self, other):
        return self.height > other.height or self.width > other.width

    def __eq__(self, other):
        return self.height == other.height and self.width == other.width

    @property
    def as_shape(self) -> Tuple[int, int]:
        """Get as numpy shape (h, w)."""
        return self.height, self.width

    @property
    def area(self) -> int:
        """Pixel area of image."""
        return self.width * self.height

    @property
    def w(self) -> int:
        """Alias for width."""
        return self.width

    @w.setter
    def w(self, w):
        """Set width."""
        self.width = w

    @property
    def h(self) -> int:
        """Alias for height."""
        return self.height

    @h.setter
    def h(self, h):
        """Set height."""
        self.height = h

    @classmethod
    def from_np(cls, np_array: np.ndarray) -> Size:
        """Create instance from numpy array.

        Args:
            np_array: Numpy image array
        """
        size = cls()
        size.height, size.width = np_array.shape[:2]
        return size

    @classmethod
    def calculate_new(
        cls, orig_size: Size, scale: float = None, new_size: Size = None,
    ) -> Size:
        """Calculate new dimensions and maintain image aspect ratio.

        Scale is given precedence over new size dims.

        Args:
            orig_size: Size object of original file dims
            scale: Optional factor to scale by (0-1.0)
            new_size: Optional Size object of desired new dims

        Returns: Size object of correct proprotions for new size
        """
        calc_size = Size()
        LOG.info("Calculating size for original: %s", orig_size)
        if scale is not None and scale > 0.0:
            LOG.info("Scaling image by %f", scale)
            calc_size.width = int(round(orig_size.width * scale))
            calc_size.height = int(round(orig_size.height * scale))
            LOG.info("New size: %s", calc_size)
            return calc_size
        if new_size is not None:
            if new_size.width > 0 and new_size.height > 0:
                LOG.info("Both width and height provided; ")
                calc_size = new_size
            elif new_size.width > 0 and new_size.height <= 0:
                LOG.info("Calculating height based on width")
                calc_size.width = new_size.width
                calc_size.height = int(
                    round((calc_size.width * orig_size.height) / orig_size.width)
                )
            elif new_size.height > 0 and new_size.width <= 0:
                LOG.info("Calculating width based on height")
                calc_size.height = new_size.height
                calc_size.width = int(
                    round((calc_size.height * orig_size.width) / orig_size.height)
                )
            LOG.info("New size: %s", calc_size)
            return calc_size
        LOG.info("No new width, height, or pct scale supplied; using current dims")
        return orig_size
