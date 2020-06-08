"""Resize and validation functions.

Based on original code from:
https://github.com/VingtCinq/python-resize-image
"""
import logging
import math
import sys
from functools import wraps
from typing import Tuple, Union
import numpy as np
import cv2

from PIL import Image
from PIL.Image import Image as PILImage

from pyimgtool.data_structures import Size
from pyimgtool.commands.mat import create_mat
from pyimgtool.exceptions import ImageTooSmallError, ResizeNotNeededError

LOG = logging.getLogger(__name__)


def validate(validator):
    """Return a decorator that validates arguments with provided `validator` function.

    This will also store the validator function as `func.validate`.
    The decorator returned by this function can bypass the validator
    if `validate=False` is passed as argument; otherwise, the function is
    called directly.

    The validator must raise an exception if the function can not
    be called.
    """

    def decorator(func):
        """Bound decorator to a particular validator function."""

        @wraps(func)
        def wrapper(
            im: Union[np.ndarray, PILImage], size: Size, validate=True, *args, **kwargs
        ):
            if validate:
                validator(im, size)
            return func(im, size, *args, **kwargs)

        return wrapper

    return decorator


def is_big_enough(im: Union[np.ndarray, PILImage], size: Size) -> None:
    """Check that the image's size superior to `size`.

    Args:
        im: Numpy array or PIL image
        size: Size object

    Raises:
        ImageTooSmallError: if image size < `size`
    """
    h, w = im.shape[:2] if type(im) == np.ndarray else im.size[::-1]
    if (size.w > w) and (size.h > h):
        raise ImageTooSmallError((w, h), size)


def width_is_big_enough(im: Union[np.ndarray, PILImage], size: Size) -> None:
    """Check that the image width is greater than desired width.

    Args:
        im: Numpy array
        size: Size object

    Raises:
        ImageTooSmallError: if image width < size[0]
    """
    im_width = im.shape[1] if type(im) == np.ndarray else im.width
    if size.width > im_width:
        raise ImageTooSmallError(im_width, size.width)


def height_is_big_enough(im: Union[np.ndarray, PILImage], size: Size) -> None:
    """Check that the image height is greater than desired height.

    Args:
        im: Numpy array or PIL Image
        size: Tuple of width, height.

    Raises:
        ImageTooSmallError: if image height < size.height.
    """
    im_height = im.shape[0] if type(im) == np.ndarray else im.height
    if size.height > im_height:
        raise ImageTooSmallError(im_height, size.height)


@validate(is_big_enough)
def resize_crop(image: PILImage, size: Size) -> PILImage:
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


@validate(is_big_enough)
def resize_crop_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> np.ndarray:
    """Crop the image with a centered rectangle of the specified size.

    Args:
        image: Numpy array image
        size: Size object representing desired new size

    Returns: Numpy array image, resized
    """
    orig = Size.from_np(im)
    if orig > size:
        LOG.debug("Resizing before crop")
        method = "width" if size.w < size.h else "height"
        im = resize_opencv(method, im, size, resample=resample)
        orig = Size.from_np(im)
    left = (orig.w - size.w) / 2
    top = (orig.h - size.h) / 2
    right = orig.w - left
    bottom = orig.h - top
    box = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    LOG.debug("Crop box size: %s", box)
    crop = im[box[1] : box[3], box[0] : box[2]]
    return crop


# @validate(is_big_enough)
def resize_cover(image: PILImage, size: Size, resample=Image.LANCZOS) -> PILImage:
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
    ratio = max((size.w / img_size.w), (size.h / img_size.h))
    new_size = (
        int(math.ceil(img_size.w * ratio)),
        int(math.ceil(img_size.h * ratio)),
    )
    img = img.resize(new_size, resample)
    img = resize_crop(img, size)
    img.format = img_format
    return img


# @validate(is_big_enough)
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
    image: PILImage,
    size: Size,
    resample=Image.LANCZOS,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
    bg_size: Size = None,
) -> PILImage:
    """Resize image to fill specified area.

    Image is not cropped and aspect ratio is kept intact.
    Same behavior as `background-size: contain`.

    Args:
        image: Pillow image instance
        size: Size object
        resample: Resample method
        bg_color: RGBA Tuple for background (if image smaller than `size`)
        bg_size: Background size (if different from `size`)

    Returns: PIL Image
    """
    img_format = image.format
    img = resize_thumbnail(image, size, resample=resample, validate=False)
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
    im = resize_thumbnail_opencv(im, size, resample=resample, validate=False)
    if bg_size is None:
        bg_size = size
    h, w, c = im.shape
    mat = 255 * np.ones((size.h, size.w, c), dtype=np.uint8)
    xx = int(math.ceil((size.w - w) / 2))
    yy = int(math.ceil((size.h - h) / 2))
    mat[yy : yy + h, xx : xx + w] = im
    return mat.copy()


@validate(width_is_big_enough)
def resize_width(image: PILImage, size: Size, resample=Image.LANCZOS) -> PILImage:
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
    size.height = int(max((size.width / img.width) * img.height, 1))
    LOG.debug("resize_width() with size: %s", size)
    img.thumbnail(size, resample)
    img.format = img_format
    return img


@validate(width_is_big_enough)
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
    orig_h, orig_w = im.shape[:2]
    r = size.width / float(orig_w)
    dim = size.width, int(orig_h * r)
    return cv2.resize(im, dim, interpolation=resample)


@validate(height_is_big_enough)
def resize_height(image: PILImage, size: Size, resample=Image.LANCZOS) -> PILImage:
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
    height = size.height
    new_width = int(math.ceil((height / img.height) * image.width))
    img.thumbnail((new_width, height), resample)
    img.format = img_format
    return img


@validate(height_is_big_enough)
def resize_height_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> np.ndarray:
    """Resize image to according to specified height.

    Aspect ratio is kept intact.

    Args:
        im: Numpy array
        size: Size object of desired size
        resample: Resample method

    Returns: Numpy array
    """
    orig_h, orig_w = im.shape[:2]
    r = size.height / float(orig_h)
    dim = int(orig_w * r), size.height
    return cv2.resize(im, dim, resample)


@validate(is_big_enough)
def resize_thumbnail(image: PILImage, size: Size, resample=Image.LANCZOS) -> PILImage:
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


@validate(is_big_enough)
def resize_thumbnail_opencv(
    im: np.ndarray, size: Size, resample=cv2.INTER_AREA
) -> PILImage:
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
    try:
        return getattr(sys.modules[__name__], method)(*args, **kwargs)
    except AttributeError:
        LOG.critical(
            f"Invalid method '{method}'; should be one of {', '.join(valid_methods)}"
        )


def get_method(
    orig_size: Size, width=None, height=None, scale=None, longest=None, shortest=None
) -> Tuple[str, Size]:
    """Determine which resize method to use based on calculated size.

    Parameters
    ----------
    width
        Width parameter from cli
    height
        Height parameter from cli
    new_size
        Size parameter from cli
    scale
        Scale parameter from cli
    longest
        Longest size parameter
    shortest
        Shortest size parameter

    Returns
    -------
    Tuple[str, Size]:
        Resize method, calculated size
    """
    resize_method = "thumbnail"
    new_size = Size(width, height)
    # if new_size == orig_size:
    #     raise ResizeNotNeededError
    if longest is not None:
        if orig_size.width >= orig_size.height:
            resize_method = "width"
            new_size.width = longest
        else:
            resize_method = "height"
            new_size.height = longest
    elif shortest is not None:
        if orig_size.width <= orig_size.height:
            resize_method = "width"
            new_size.width = shortest
        else:
            resize_method = "height"
            new_size.height = shortest
    elif width is not None and height is not None:
        resize_method = "crop"
        if new_size > orig_size:
            resize_method = "contain"
    elif width is not None and height is None:
        resize_method = "width"
    elif width is None and height is not None:
        resize_method = "height"
    else:
        new_size = Size.calculate_new(orig_size, scale, new_size,)
    new_size.width = orig_size.width if None else new_size.width
    new_size.height = orig_size.height if None else new_size.height
    if new_size == orig_size:
        raise ResizeNotNeededError
    LOG.debug(
        "Original size: %s; New size: %s; Resize method: %s",
        orig_size,
        new_size,
        resize_method,
    )
    return resize_method, new_size
