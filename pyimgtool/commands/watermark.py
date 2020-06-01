"""Watermark image with text or another image."""

import logging
from datetime import datetime
from pathlib import PurePath
from typing import Dict, Any
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont, ImageStat

from pyimgtool.commands.resize import resize_height
from pyimgtool.data_structures import Size, Position
from pyimgtool.utils import get_pkg_root

LOG = logging.getLogger(__name__)


def get_region_stats(im: Image, region: list) -> ImageStat:
    """Get ImageStat object for region of PIL image.

    Args:
        im: The image to get the luminance of
        region: The region to get the luminance of, in the form
                [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]

    Returns: ImageStat object with stats
    """
    image_l = im.convert("L")
    mask = Image.new("L", image_l.size, 0)
    drawing_layer = ImageDraw.Draw(mask)
    drawing_layer.rectangle(region, fill=255)
    return ImageStat.Stat(image_l, mask=mask)


def find_best_location(im: Image, size: Size, padding: float) -> Position:
    """Find the best location for the watermark.

    The best location is the one with least luminance variance.

    Args:
        im: PIL Image
        size: Size of watermark image
        padding: Proportion of padding to add around watermark

    Returns: Position object
    """
    bl_padding = tuple(
        map(
            lambda x: int(x),
            [padding * im.size[0], im.size[1] - size.height - padding * im.size[1]],
        )
    )
    br_padding = tuple(
        map(
            lambda x: int(x),
            [
                im.size[0] - size.width - padding * im.size[0],
                im.size[1] - size.height - padding * im.size[1],
            ],
        )
    )
    tl_padding = tuple(
        map(lambda x: int(x), [padding * im.size[0], padding * im.size[0]])
    )
    tr_padding = tuple(
        map(
            lambda x: int(x),
            [im.size[0] - size.width - padding * im.size[0], padding * im.size[1]],
        )
    )
    bc_padding = tuple(
        map(
            lambda x: int(x),
            [
                im.size[0] / 2 - size.width / 2,
                im.size[1] - size.height - padding * im.size[1],
            ],
        )
    )
    paddings = [bl_padding, br_padding, tl_padding, tr_padding, bc_padding]
    vars = list(
        map(
            lambda padding: get_region_stats(
                im, [padding, (padding[0] + size.width, padding[1] + size.height)]
            ).stddev[0],
            paddings,
        )
    )
    minimum = min(vars)
    index = vars.index(minimum)
    locations = [
        Position.BOTTOM_LEFT,
        Position.BOTTOM_RIGHT,
        Position.TOP_LEFT,
        Position.TOP_RIGHT,
        Position.BOTTOM_CENTER,
    ]
    return locations[index]


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
    im: Image,
    watermark_image: Image,
    scale: float = 0.2,
    position: Position = None,
    opacity: float = 0.3,
    padding: int = 10,
) -> Image:
    """Watermark with image according to Config.

    Args:
        im: PIL Image
        watermark_image: PIL Image
        scale: Scale for watermark relative to image
        position: Position of watermark
        opacity: Watermark layer opacity from 0 to 1
        padding: Pixels of padding for watermark

    Returns: Watermarked image
    """
    watermark_image = watermark_image.convert("RGBA")
    LOG.info("Watermark: %s", watermark_image.size)
    watermark_ratio = watermark_image.height / im.height
    LOG.info("Watermark size ratio: %.4f", watermark_ratio)
    if watermark_ratio > scale:
        LOG.debug(
            "Resizing watermark from %.4f to %.4f scale", watermark_ratio, scale,
        )
        watermark_image = resize_height(
            watermark_image, (int(im.width * scale), int(im.height * scale)),
        )
        LOG.debug("New watermark dims: %s", watermark_image.size)
    offset_x = padding
    offset_y = padding
    watermark_size = Size(watermark_image.width, watermark_image.height)
    mask = watermark_image.split()[3].point(lambda i: i * opacity)
    pos = (
        (im.width - watermark_image.width - offset_x),
        (im.height - watermark_image.height - offset_y),
    )
    loc = find_best_location(im, watermark_size, 0.05)
    LOG.debug("Best detected watermark loc: %s", loc)
    im.paste(watermark_image, pos, mask)
    return im


def with_image_opencv(
    im: np.ndarray,
    watermark_image: np.ndarray,
    scale: float = 0.2,
    position: Position = None,
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
    wH, wW = watermark_image.shape[:2]
    new_size = Size.calculate_new(Size(wW, wH), scale)
    watermark_image = cv2.resize(
        watermark_image, (new_size.width, new_size.height), interpolation=cv2.INTER_AREA
    )
    wH, wW = watermark_image.shape[:2]
    h, w = im.shape[:2]
    im = np.dstack([im, np.ones((h, w), dtype="uint8") * 255])
    # construct an overlay that is the same size as the input
    # image, (using an extra dimension for the alpha transparency),
    # then add the watermark to the overlay in the bottom-right
    # corner
    overlay = np.zeros((h, w, 4), dtype="uint8")
    overlay[h - wH - 10 : h - 10, w - wW - 10 : w - 10] = watermark_image
    # blend the two images together using transparent overlays
    output = im.copy()
    cv2.addWeighted(overlay, opacity, output, 1.0, 0, output)
    return output


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
            PurePath.joinpath(get_pkg_root(), "fonts", "SourceSansPro-Regular.ttf")
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
        im, [offset_x, offset_y, text_width, im.height - text_height]
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
