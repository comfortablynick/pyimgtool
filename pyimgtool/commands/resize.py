"""Resize and validation functions.

Based on original code from:
https://github.com/VingtCinq/python-resize-image
"""
import logging
import math
import sys
from functools import wraps
from typing import Tuple
import numpy as np
import cv2

from PIL import Image

from pyimgtool.data_structures import Size
from pyimgtool.exceptions import ImageSizeError

LOG = logging.getLogger(__name__)


def validate(validator):
    """Return a decorator that validates arguments with provided `validator` function.

    This will also store the validator function as `func.validate`.
    The decorator returned by this function, can bypass the validator
    if `validate=False` is passed as argument otherwise the fucntion is
    called directly.

    The validator must raise an exception, if the function can not
    be called.
    """

    def decorator(func):
        """Bound decorator to a particular validator function."""

        @wraps(func)
        def wrapper(image, size, validate=True):
            if validate:
                validator(image, size)
            return func(image, size)

        return wrapper

    return decorator


def _is_big_enough(image: Image, size: Tuple[int, int]):
    """Check that the image's size superior to `size`."""
    if (size[0] > image.size[0]) and (size[1] > image.size[1]):
        raise ImageSizeError(image.size, size)


def _width_is_big_enough(image: Image, width: int):
    """Check that the image width is superior to `width`."""
    if width > image.size[0]:
        raise ImageSizeError(image.size[0], width)


def _height_is_big_enough(image: Image, size: Tuple[int, int]) -> None:
    """Check that the image height is greater than desired height.

    Args:
        image: PIL image.
        size: Tuple of width, height.

    Raises:
        ImageSizeError: if image height < size[1].
    """
    if size[1] > image.size[1]:
        raise ImageSizeError(image.size[1], size[1])


@validate(_is_big_enough)
def resize_crop(image: Image, size: Size) -> Image:
    """Crop the image with a centered rectangle of the specified size.

    Args:
        image: Pillow image instance
        size: Size object representing desired new size

    Returns: PIL Image
    """
    img_format = image.format
    image = image.copy()
    orig_w, orig_h = image.size
    LOG.debug(f"{size}")
    left = (orig_w - size.w) / 2
    top = (orig_h - size.h) / 2
    right = orig_w - left
    bottom = orig_h - top
    rect = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    LOG.debug("New rect size: %s", rect)
    left, top, right, bottom = rect
    crop = image.crop(rect)
    crop.format = img_format
    return crop


@validate(_is_big_enough)
def resize_cover(image: Image, size: Tuple[int, int], resample=Image.LANCZOS) -> Image:
    """Resize image to fill the specified area; crop as needed.

    Same behavior as `background-size: cover`.

    Args:
        image: Pillow image instance
        size: 2-Tuple of image dimensions (width, height)
        resample: Resample method

    Returns: PIL Image
    """
    img_format = image.format
    img = image.copy()
    img_size = img.size
    ratio = max(size[0] / img_size[0], size[1] / img_size[1])
    new_size = (
        int(math.ceil(img_size[0] * ratio)),
        int(math.ceil(img_size[1] * ratio)),
    )
    img = img.resize(new_size, resample)
    img = resize_crop(img, size)
    img.format = img_format
    return img


def resize_contain(
    image: Image,
    size: Size,
    resample=Image.LANCZOS,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
    bg_size: Size = None,
) -> Image:
    """Resize image to fill specified area.

    Image is not cropped and aspect ratio is kept intact.
    Same behavior as `background-size: contain`.

    Args:
        image: Pillow image instance
        size: 2-Tuple of image dimensions (width, height)
        resample: Resample method
        bg_color: RGBA Tuple for background (if image smaller than `size`)
        bg_size: Background size (if different from `size`)

    Returns: PIL Image
    """
    img_format = image.format
    img = image.copy()
    img.thumbnail(size, resample)
    if bg_size is None:
        bg_size = Size(*size)
    background = Image.new("RGBA", (bg_size.width, bg_size.height), bg_color)
    img_position = (
        int(math.ceil((bg_size.w - img.size[0]) / 2)),
        int(math.ceil((bg_size.h - img.size[1]) / 2)),
    )
    background.paste(img, img_position)
    background.format = img_format
    return background.convert("RGB")


@validate(_width_is_big_enough)
def resize_width(image: Image, size: Size, resample=Image.LANCZOS) -> Image:
    """Resize image to according to specified width.

    Aspect ratio is kept intact.

    Args:
        image: Pillow image instance
        size: Size object of desired size
        resample: Resample method

    Returns: PIL Image
    """
    img_format = image.format
    img = image.copy()
    width = size.width
    # If the origial image has already the good width, return it
    if img.width == width:
        return image
    new_height = int(math.ceil((width / img.width) * img.height))
    img.thumbnail((width, new_height), resample)
    img.format = img_format
    return img


@validate(_height_is_big_enough)
def resize_height(image: Image, size: Size, resample=Image.LANCZOS) -> Image:
    """Resize image to according to specified height.

    Aspect ratio is kept intact.

    Args:
        image: Pillow image instance
        size: Size object of desired size
        resample: Resample method

    Returns: PIL Image
    """
    img_format = image.format
    img = image.copy()
    # If the origial image has already the good height, return it
    height = size.height
    if img.height == height:
        return image
    new_width = int(math.ceil((height / img.height) * image.width))
    img.thumbnail((new_width, height), resample)
    img.format = img_format
    return img


def resize_thumbnail(image: Image, size: Size, resample=Image.LANCZOS) -> Image:
    """Resize image to according to specified size.

    Aspect ratio is kept intact while trying best to match `size`.

    Args:
        image: Pillow image instance
        size: Size object of desired size
        resample: Resample method

    Returns: PIL Image
    """
    image.thumbnail(size, resample)
    return image


def resize(method, *args, **kwargs):
    """Direct arguments to one of the resize functions.

    Args:
        method: One among 'crop', 'cover', 'contain', 'width', 'height' or 'thumbnail'
        image: Pillow image instance
        size: Size object
    """
    valid_methods = ["crop", "cover", "contain", "width", "height", "thumbnail"]
    if method not in valid_methods:
        raise ValueError(
            f"method argument should be one of: {', '.join([ repr(m) for m in valid_methods])}"
        )
    method = f"resize_{method}"
    LOG.info("Resizing with %s()", method)
    return getattr(sys.modules[__name__], method)(*args, **kwargs)


def resize_opencv(im: np.ndarray, size: Size, resample=cv2.INTER_AREA):
    """Resize with opencv, keeping ratio intact.

    Args:
        im: Numpy array
        size: Image size
        resample: Resampling interpolation algorithm

    Returns: Resized image array
    """
    LOG.debug("Resizing to %s", size)
    im = cv2.resize(im, (size.width, size.height), interpolation=resample)
    return im
