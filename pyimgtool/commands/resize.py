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
from pyimgtool.commands.mat import create_mat
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


def check(validator):
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
        def wrapper(im, size, validate=True):
            if validate:
                validator(im, size)
            return func(im, size)

        return wrapper

    return decorator


def is_big_enough(im: np.ndarray, size: Size) -> None:
    """Check that the image's size superior to `size`.

    Args:
        im: Numpy array containing image
        size: Size object

    Raises:
        ImageSizeError: if image size < `size`
    """
    h, w = im.shape[:2]
    if (size.w > w) and (size.h > h):
        raise ImageSizeError((w, h), size)


def _is_big_enough(image: Image, size: Size) -> None:
    """Check that the image's size superior to `size`.

    Args:
        image: PIL Image
        size: Size object

    Raises:
        ImageSizeError: if image size < `size`
    """
    if (size.w > image.size[0]) and (size.h > image.size[1]):
        raise ImageSizeError(image.size, size)


def width_is_big_enough(im: np.ndarray, size: Size) -> None:
    """Check that the image width is greater than desired width.

    Args:
        im: Numpy array
        size: Size object

    Raises:
        ImageSizeError: if image width < size[0]
    """
    if size.width > (im_width := im.shape[1]):
        raise ImageSizeError(im_width, size.width)


def _width_is_big_enough(image: Image, size: Size) -> None:
    """Check that the image width is greater than desired width.

    Args:
        image: PIL image.
        size: Size object

    Raises:
        ImageSizeError: if image width < size[0].
    """
    if (width := size.width) > image.size[0]:
        raise ImageSizeError(image.size[0], width)


def height_is_big_enough(im: np.ndarray, size: Size) -> None:
    """Check that the image height is greater than desired height.

    Args:
        im: Numpy array
        size: Tuple of width, height.

    Raises:
        ImageSizeError: if image height < size[1].
    """
    if size.height > (im_height := im.shape[0]):
        raise ImageSizeError(im_height, size.height)


def _height_is_big_enough(image: Image, size: Size) -> None:
    """Check that the image height is greater than desired height.

    Args:
        image: PIL image.
        size: Tuple of width, height.

    Raises:
        ImageSizeError: if image height < size[1].
    """
    if (height := size.height) > image.size[1]:
        raise ImageSizeError(image.size[1], height)


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
    left = (orig_w - size.w) / 2
    top = (orig_h - size.h) / 2
    right = orig_w - left
    bottom = orig_h - top
    rect = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    LOG.debug("New rect size: %s", rect)
    crop = image.crop(rect)
    crop.format = img_format
    return crop


@check(is_big_enough)
def resize_crop_opencv(im: np.ndarray, size: Size) -> np.ndarray:
    """Crop the image with a centered rectangle of the specified size.

    Args:
        image: Numpy array image
        size: Size object representing desired new size

    Returns: Numpy array image, resized
    """
    orig_h, orig_w = im.shape[:2]
    left = (orig_w - size.w) / 2
    top = (orig_h - size.h) / 2
    right = orig_w - left
    bottom = orig_h - top
    box = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    LOG.debug("Crop box size: %s", box)
    crop = im[box[1] : box[3], box[0] : box[2]]
    return crop


@validate(_is_big_enough)
def resize_cover(image: Image, size: Size, resample=Image.LANCZOS) -> Image:
    """Resize image to fill the specified area; crop as needed.

    Same behavior as `background-size: cover`.

    Args:
        image: Pillow image instance
        size: Size object of image dimensions
        resample: Resample method

    Returns: PIL Image
    """
    img_format = image.format
    img = image.copy()
    img_size = Size(img.size)
    ratio = max(size.w / img_size.w, size.h / img_size.h)
    new_size = (
        int(math.ceil(img_size.w * ratio)),
        int(math.ceil(img_size.h * ratio)),
    )
    img = img.resize(new_size, resample)
    img = resize_crop(img, size)
    img.format = img_format
    return img


@check(is_big_enough)
def resize_cover_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> np.ndarray:
    """Resize image to fill the specified area; crop as needed.

    Same behavior as `background-size: cover`.

    Args:
        im: Numpy array
        size: Size object of image dimensions
        resample: Resample method

    Returns: Numpy array, resized
    """
    img_format = im.format
    orig_size = Size.from_np(im)
    ratio = max(size.w / orig_size.w, size.h / orig_size.h)
    new_size = (
        int(math.ceil(orig_size.w * ratio)),
        int(math.ceil(orig_size.h * ratio)),
    )
    img = create_mat(im, new_size)
    img = resize_crop_opencv(img, size)
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
        bg_size = size
    background = Image.new("RGBA", tuple(bg_size), bg_color)
    img_position = (
        int(math.ceil((bg_size.w - img.size[0]) / 2)),
        int(math.ceil((bg_size.h - img.size[1]) / 2)),
    )
    background.paste(img, img_position)
    background.format = img_format
    return background.convert("RGB")


def resize_contain_opencv(
    im: np.ndarray,
    size: Size,
    resample=cv2.INTER_AREA,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
    bg_size: Size = None,
) -> np.ndarray:
    """Resize image to fill specified area.

    Image is not cropped and aspect ratio is kept intact.
    Same behavior as `background-size: contain`.

    Args:
        image: Numpy array
        size: Size object
        resample: Resample method
        bg_color: RGBA Tuple for background (if image smaller than `size`)
        bg_size: Background size (if different from `size`)

    Returns: Numpy array
    """
    im = resize_thumbnail_opencv(im, size)
    if bg_size is None:
        bg_size = size
    h, w, c = im.shape
    mat = 255 * np.ones((size.h, size.w, c), dtype=np.uint8)
    xx = (size.w - w) // 2
    yy = (size.h - h) // 2
    mat[yy : yy + h, xx : xx + w] = im
    return mat.copy()


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
    # If the origial image has already the good width, return it
    if img.width == size.width:
        return image
    size.height = int(math.ceil((size.width / img.width) * img.height))
    LOG.debug("resize_width() with size: %s", size)
    img.thumbnail(size, resample)
    img.format = img_format
    return img


@check(width_is_big_enough)
def resize_width_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> np.ndarray:
    """Resize image to according to specified width.

    Aspect ratio is kept intact.

    Args:
        im: Numpy array
        size: Size object of desired size
        resample: Resample method

    Returns: Numpy array
    """
    LOG.debug("Resample: %d", resample)
    orig_h, orig_w = im.shape[:2]
    # If the origial image has already the good width, return it
    if orig_w == size.width:
        return im
    new_height = int(math.ceil((size.width / orig_w) * orig_h))
    return cv2.resize(im, (size.width, new_height), interpolation=resample)


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


@check(height_is_big_enough)
def resize_height_opencv(im: np.ndarray, size: Size, resample=cv2.INTER_AREA) -> Image:
    """Resize image to according to specified height.

    Aspect ratio is kept intact.

    Args:
        im: Numpy array
        size: Size object of desired size
        resample: Resample method

    Returns: Numpy array
    """
    orig_h, orig_w = im.shape[:2]
    # If the origial image has already the good height, return it
    height = size.height
    if orig_h == height:
        return im
    new_width = int(math.ceil((height / orig_h) * orig_w))
    return cv2.resize(im, (new_width, height), resample)


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


def resize_thumbnail_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> Image:
    """Resize image to according to specified size.

    Aspect ratio is kept intact while trying best to match `size`.

    Args:
        image: Numpy array
        size: Size object of desired size
        resample: Resample method

    Returns: Array of resized image
    """
    y, x = im.shape[:2]
    if x > size.w:
        y = int(max(y * size.w / x, 1))
        x = int(size.w)
    if y > size.h:
        x = int(max(x * size.h / y, 1))
        y = int(size.h)
    new_size = Size(x, y)
    LOG.debug("Thumbnail calculated size: %s", new_size)
    if new_size == Size.from_np(im):
        LOG.debug("Thumbnail size matches existing image size")
        return im
    return cv2.resize(im, tuple(new_size), interpolation=resample)


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


def resize_opencv(method, *args, **kwargs):
    """Direct arguments to one of the resize functions.

    Args:
        method: One among 'crop', 'cover', 'contain', 'width', 'height' or 'thumbnail'
        image: Numpy array
        size: Size object
    """
    method = f"resize_{method}_opencv"
    valid_methods = [
        x for x in globals().keys() if x.endswith("opencv") and x != "resize_opencv"
    ]
    LOG.info("Resizing with %s()", method)
    LOG.debug("resize_opencv() args: %s", kwargs)
    try:
        return getattr(sys.modules[__name__], method)(*args, **kwargs)
    except AttributeError:
        LOG.critical(
            f"Invalid method '{method}'; should be one of {', '.join(valid_methods)}"
        )


# def resize_opencv(im: np.ndarray, size: Size, resample=cv2.INTER_AREA):
#     """Resize with opencv, keeping ratio intact.
#
#     Args:
#         im: Numpy array
#         size: Image size
#         resample: Resampling interpolation algorithm
#
#     Returns: Resized image array
#     """
#     orig_size = Size.from_np(im)
#     LOG.debug("Resizing with opencv from %s to %s", orig_size, size)
#     if orig_size < size:
#         LOG.debug("Orig image smaller than desired dimensions")
#     im = cv2.resize(im, tuple(size), interpolation=resample)
#     return im
