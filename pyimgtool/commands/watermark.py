"""Watermark image with text or another image."""

import logging
import os
from datetime import datetime
from pathlib import PurePath
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont, ImageStat

from pyimgtool.commands.resize import resize_height
from pyimgtool.data_structures import Config, Context, ImageSize, Position
from pyimgtool.utils import get_pkg_root

LOG = logging.getLogger(__name__)


def get_region_stats(im: Image, region: list) -> ImageStat:
    """Get ImageStat object for region of PIL image.

    Args:
        im: The image to get the luminance of
        region: The region to get the luminance of, in the form [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]

    Returns: ImageStat object with stats
    """
    image_l = im.convert("L")
    mask = Image.new("L", image_l.size, 0)
    drawing_layer = ImageDraw.Draw(mask)
    drawing_layer.rectangle(region, fill=255)
    return ImageStat.Stat(image_l, mask=mask)


def find_best_location(im: Image, size: ImageSize, padding: float) -> Position:
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


def get_copyright_string(im: Image, text_copyright, exif: Optional[Dict] = None) -> str:
    """Extract date taken from photo to add to copyright text.

    Args:
        im: PIL Image
        text_copyright: Copyright text
        exif: Dictionary of exif data
    """
    photo_dt = datetime.now()
    if exif is not None:
        try:
            photo_dt = datetime.strptime(
                exif["Exif"]["DateTimeOriginal"].decode("utf-8"), "%Y:%m:%d %H:%M:%S",
            )
        except KeyError:
            pass
    copyright_year = photo_dt.strftime("%Y")
    LOG.info("Using copyright text: %s", text_copyright)
    LOG.info("Photo date from exif: %s", photo_dt)
    return f"© {copyright_year} {text_copyright}"


def with_image(im: Image, cfg: Config, ctx: Context) -> Image:
    """Watermark with image according to Config.

    Args:
        im: PIL Image
        cfg: Config object
        ctx: Context object

    Returns: Watermarked image
    """
    if cfg.watermark_image is None:
        LOG.error("Missing watermark_image in cfg")
        return im
    watermark_image = Image.open(os.path.expanduser(cfg.watermark_image)).convert(
        "RGBA"
    )
    LOG.info("Watermark: %s", watermark_image.size)
    watermark_ratio = watermark_image.height / im.height
    LOG.info("Watermark size ratio: %.4f", watermark_ratio)
    if watermark_ratio > cfg.watermark_scale:
        LOG.debug(
            "Resizing watermark from %.4f to %.4f scale",
            watermark_ratio,
            cfg.watermark_scale,
        )
        watermark_image = resize_height(
            watermark_image,
            (int(im.width * cfg.watermark_scale), int(im.height * cfg.watermark_scale)),
        )
        LOG.debug("New watermark dims: %s", watermark_image.size)
    offset_x = cfg.watermark_padding
    offset_y = cfg.watermark_padding
    ctx.watermark_size = ImageSize(watermark_image.width, watermark_image.height)
    mask = watermark_image.split()[3].point(lambda i: i * cfg.watermark_opacity)
    pos = (
        (im.width - watermark_image.width - offset_x),
        (im.height - watermark_image.height - offset_y),
    )
    loc = find_best_location(im, ctx.watermark_size, 0.05)
    LOG.debug("Best detected watermark loc: %s", loc)
    im.paste(watermark_image, pos, mask)
    return im


def with_text(
    im: Image,
    text: str = None,
    copyright: str = None,
    scale: float = 0.2,
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
        text: General text to add to image
        copyright: Text for copyright
        scale: Scale for size of text relative to image
        opacity: Text layer opacity from 0 to 1
        padding: Pixels of padding for text
        exif: Image metadata

    Return: Watermarked image
    """
    if text is None and copyright is None:
        LOG.error("Need either text or copyright text")
        return im
    if copyright is not None:
        # Add date photo taken to copyright text
        text = get_copyright_string(im, copyright, exif)
    layer = Image.new("RGBA", (im.width, im.height), (255, 255, 255, 0))

    font_size = 1  # starting size
    offset_x = padding
    offset_y = padding

    try:
        # cwd = PurePath(os.path.dirname(__file__))
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
