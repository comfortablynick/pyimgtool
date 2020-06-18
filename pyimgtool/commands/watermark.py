"""Watermark image with text or another image."""

import logging
from datetime import datetime
from pathlib import PurePath
from pprint import pformat
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
from PIL.Image import Image as PILImage

from pyimgtool import utils
from pyimgtool.commands.resize import resize_height
from pyimgtool.data_structures import Box, Position, Size, Stat
from pyimgtool.exceptions import OverlaySizeError

LOG = logging.getLogger(__name__)


def get_region_stats(im: PILImage, region: Box) -> ImageStat:
    """Get ImageStat object for region of PIL image.

    Parameters
    ----------
    im : PIL Image
        The image to get the luminance of
    region : Box
        Coordinates for region

    Returns: ImageStat object with stats
    """
    LOG.debug("Region for stats: %s", region)
    m = Image.new("L", im.size, 0)
    drawing_layer = ImageDraw.Draw(m)
    drawing_layer.rectangle(tuple(region), fill=255)
    st = ImageStat.Stat(im, mask=m)
    return Stat(stddev=st.stddev[0], mean=st.mean[0])


def get_region_stats_np(im: np.ndarray, region: Box) -> Stat:
    """Get array region stats using Numpy.

    Parameters
    ----------
    im : np.ndarray
        Input image to analyze
    region : Box
        Coordinates for region

    Returns
    -------
    Stat
        Stat object containing various statistics of region
    """
    x0, y0, x1, y1 = region
    dtype = np.float64
    im = im[y0:y1, x0:x1].copy()
    stddev = np.std(im, dtype=dtype)
    mean = np.mean(im, dtype=dtype)
    distance = abs(mean - 128) / 128.0
    weighted_dev = stddev - (stddev * distance)
    return Stat(stddev=stddev, mean=mean, weighted_dev=weighted_dev)


def find_best_location(
    im: Image, size: Size, padding: float
) -> Tuple[Position, Box, Stat]:
    """Find the best location for the watermark.

    The best location is the one with least luminance variance.

    Args:
        im: PIL Image
        size: Size of watermark image
        padding: Proportion of padding to add around watermark

    Returns: Position object
    """
    im_size = Size(*im.size)
    positions = []
    for p in Position:
        if p is not Position.CENTER:
            pos = p.calculate_for_overlay(im_size, size, padding)
            st = get_region_stats(im, pos)
            positions.append((p, pos, st))
    LOG.debug("Positions: %s", pformat(positions))
    # utils.show_image_cv2(im)
    return min(positions, key=lambda i: i[2].stddev)


def find_best_position(
    im: np.ndarray, size: Size, padding: float
) -> Tuple[Position, Box, Stat]:
    """Find the best location for the watermark.

    The best location is the one with lowest luminance stddev.

    Parameters
    ----------
    im
        Image array
    size
        Size of watermark image
    padding
        Proportion of padding to add around watermark

    Returns
    -------
    Position, Box, Stat
    """
    im_size = Size.from_np(im)
    positions = []
    for p in Position:
        if p is not Position.CENTER:
            pos = p.calculate_for_overlay(im_size, size, padding)
            st = get_region_stats_np(im, pos)
            positions.append((p, pos, st))
    LOG.debug("Positions: %s", pformat(positions))
    # utils.show_image_cv2(im)
    return min(positions, key=lambda i: i[2].stddev)


def get_copyright_string(exif: Dict[Any, Any]) -> str:
    """Extract date taken from photo to add to copyright text.

    Args:
        exif: Dictionary of exif data

    Returns: string of copyright symbol and date
    """
    if exif is not None:
        # LOG.debug("Exif data: %s", exif["Exif"])
        try:
            photo_dt = datetime.strptime(
                exif["Exif"]["DateTimeOriginal"].decode("utf-8"), "%Y:%m:%d %H:%M:%S",
            )
        except KeyError:
            photo_dt = datetime.now()
    copyright_year = photo_dt.strftime("%Y")
    # LOG.info("Using copyright text: %s", text_copyright)
    LOG.info("Photo date from exif: %s", photo_dt)
    return f"© {copyright_year}"


def with_image(
    im: PILImage,
    watermark_image: PILImage,
    scale: float = None,
    position: Position = None,
    opacity: float = 0.3,
    padding: float = 0.05,
    invert: bool = False,
) -> PILImage:
    """Watermark with image according to Config.

    Parameters
    ----------
    im
        PIL Image
    watermark_image
        PIL Image
    scale
        Scale for watermark size
    position
        Position of watermark
    opacity
        Watermark layer opacity from 0 to 1
    padding
        Proportion of watermark image to use as padding
    invert
        Invert watermark image

    Returns
    -------
    PIL Image with watermark
    """
    watermark_image = watermark_image.convert("RGBA")
    LOG.info("Watermark: %s", watermark_image.size)
    if scale is not None and scale < 1:
        watermark_image = resize_height(
            watermark_image,
            Size(
                int(watermark_image.width * scale), int(watermark_image.height * scale)
            ),
        )
        LOG.debug("New watermark dims: %s", watermark_image.size)
    watermark_size = Size(*watermark_image.size)
    mask = watermark_image.split()[3].point(lambda i: i * opacity)
    im_gray = im.convert("L")
    if position is None:
        position, bx, stat = find_best_location(im_gray, watermark_size, padding)
        LOG.debug("Best detected watermark loc: %s", position)
    else:
        LOG.debug("Position from args: %s", position)
        bx = position.calculate_for_overlay(Size(*im.size), watermark_size, padding)
    im.paste(watermark_image, bx[:2], mask)
    return im


@utils.Log(LOG)
def with_image_opencv(
    im: np.ndarray,
    watermark_image: np.ndarray,
    scale: float = 0.2,
    position: Position = Position.BOTTOM_RIGHT,
    opacity: float = 0.3,
    padding: int = 10,
) -> Image:
    """Watermark with image according to Config.

    Args:
        im: Numpy array
        watermark_image: Numpy array
        scale: Scale for watermark relative to image
        position: Position of watermark
        opacity: Watermark layer opacity from 0 to 1
        padding: Pixels of padding for watermark

    Returns: Watermarked image array
    """
    LOG.info("Inserting watermark at position: %s", position)
    orig_im_type = im.dtype
    new_size = Size.calculate_new(Size.from_np(watermark_image), scale)
    watermark_image = cv2.resize(
        watermark_image, tuple(new_size), interpolation=cv2.INTER_AREA
    )
    watermark_image = cv2.copyMakeBorder(
        watermark_image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    wH, wW = watermark_image.shape[:2]
    h, w = im.shape[:2]
    im = np.dstack([im, np.ones((h, w), dtype=im.dtype)])
    overlay = np.zeros((h, w, 4), dtype=im.dtype)
    ww, hh, _, _ = position.calculate_for_overlay(Size(w, h), Size(wW, wH))
    LOG.debug("hh: %d, ww: %d", hh, ww)
    overlay[hh : hh + wH, ww : ww + wH] = watermark_image
    output = im.copy()
    cv2.addWeighted(overlay, opacity, output, 1.0, 0, output)
    return output.astype(orig_im_type)


def overlay_transparent(
    background: np.ndarray,
    overlay: np.ndarray,
    scale: float = None,
    position: Position = None,
    padding: float = 0.05,
    alpha: float = 0.3,
    invert: bool = False,
) -> np.ndarray:
    """Blend an image with an overlay (e.g., watermark).

    Parameters
    ----------
    background
        Main image
    overlay
        Image to blend on top of `background`
    position
        Location of overlay
    alpha
        Blend opacity, from 0 to 1
    invert
        Invert overlay image

    Returns
    -------
    Image

    Raises
    ------
    OverlaySizeError
        If overlay image is larger than background image
    """
    bg_h, bg_w = background.shape[:2]
    if scale is not None:
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
    LOG.debug("Overlay shape: %s", overlay.shape)
    h, w, c = overlay.shape
    LOG.debug(
        "Calculated margin for overlay: %s", Size(*[int(i * padding) for i in (w, h)])
    )
    bg_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    # utils.show_image_cv2(bg_gray)
    bg_gray = bg_gray.astype(np.float64)
    if position is None:
        pos, bx, stat = find_best_position(bg_gray, Size(w, h), padding)
        LOG.debug("Best calculated position: %s=%s, %s", pos, bx, stat)
    else:
        bx = position.calculate_for_overlay(
            Size.from_np(background), Size.from_np(overlay), padding
        )
        stat = get_region_stats_np(bg_gray, bx)
        LOG.debug("Position from args: %s=%s, %s", position, bx, stat)
    if (bx.x1 - bx.x0) > bg_w or (bx.y1 - bx.y0) > bg_h:
        message = f"Overlay size of {Size(w, h)} is too large for image size {Size(bg_w, bg_h)}"
        LOG.error("%s; this should be unreachable", message)
        raise OverlaySizeError(message)
    if c == 3:
        shape = h, w, 1
        LOG.debug("Adding alpha channel for overlay of shape: %s", shape)
        overlay = np.concatenate(
            [overlay, np.ones(shape, dtype=overlay.dtype) * 255], axis=2,
        )
    overlay_image = overlay[..., :3]
    mask = overlay_image / 256.0 * alpha

    invert_overlay = stat.mean > 128.0
    # Combine images, inverting overlay if necessary
    if invert_overlay:
        LOG.debug("Inverting based on luminance: %s", stat.mean)
        overlay_image = ~overlay_image
    if invert:
        # Invert whether or not we automatically inverted
        overlay_image = ~overlay_image
    background[bx.y0 : bx.y1, bx.x0 : bx.x1] = (1.0 - mask) * background[
        bx.y0 : bx.y1, bx.x0 : bx.x1
    ] + mask * overlay_image
    return background


def with_text(
    im: Image,
    text: str = None,
    copyright: bool = False,
    scale: float = 0.2,
    position: Position = None,
    opacity: float = 0.3,
    padding: int = 10,
    exif: dict = None,
) -> Image:
    """Watermark with text if program option is supplied.

    If text is equal to 'copyright', the exif data will be read
    (if available) to attempt to determine copyright date based on
    date photo was taken.

    Args:
        im: PIL Image
        text: Text to add to image
        copyright: Precede text with copyright info
        scale: Scale for size of text relative to image
        position: Text position in image
        opacity: Text layer opacity from 0 to 1
        padding: Pixels of padding for text
        exif: Image metadata

    Return: Watermarked image
    """
    if copyright and exif is not None:
        # Add date photo taken to copyright text
        text = f"{get_copyright_string(exif)} {text}"
    layer = Image.new("RGBA", (im.width, im.height), (255, 255, 255, 0))

    font_size = 1  # starting size
    offset_x = padding
    offset_y = padding

    try:
        font_path = str(
            PurePath.joinpath(
                utils.get_pkg_root(), "fonts", "SourceSansPro-Regular.ttf"
            )
        )
        font = ImageFont.truetype(font=font_path, size=font_size)
    except OSError:
        LOG.error("Could not find font '%s', aborting text watermark", font_path)
        return im

    LOG.debug("Found font '%s'", font_path)
    while font.getsize(text)[0] < scale * im.width:
        # iterate until text size is >= text_scale
        font_size += 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    if font.getsize(text)[0] > scale * im.width:
        font_size -= 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    text_width, text_height = font.getsize(text)
    LOG.debug(
        "Final text dims: %d x %d px; Font size: %d", text_width, text_height, font_size
    )
    # TODO: calculate watermark dims accurately
    stats = get_region_stats(
        im, Box(offset_x, offset_y, text_width, im.height - text_height)
    )
    LOG.debug("Region luminance: %f", stats.mean[0])
    LOG.debug("Region luminance stddev: %f", stats.stddev[0])
    d = ImageDraw.Draw(layer)
    opacity = int(round((opacity * 255)))
    LOG.info("Text opacity: %d/255", opacity)
    text_fill = 255, 255, 255, opacity
    if stats.mean[0] / 256 >= 0.5:
        text_fill = 0, 0, 0, opacity

    d.text((offset_x, offset_y), text, font=font, fill=text_fill)

    out = Image.alpha_composite(im.convert("RGBA"), layer)

    return out.convert("RGB")
