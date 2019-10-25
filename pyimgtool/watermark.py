"""Watermark image with text or another image."""

import logging
import os
from pathlib import PurePath
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageStat

from pyimgtool import resize
from pyimgtool.data_structures import Config, Context, ImageSize

LOG = logging.getLogger(__name__)


def with_image(im: Image, cfg: Config, ctx: Context) -> Image:
    """Watermark with image according to Config.

    Parameters
    ----------
    - `im` PIL Image
    - `cfg` Config object
    - `ctx` Context object

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
        watermark_image = resize.resize_height(
            watermark_image,
            (int(im.width * cfg.watermark_scale), int(im.height * cfg.watermark_scale)),
        )
        LOG.debug("New watermark dims: %s", watermark_image.size)
    ctx.watermark_size = ImageSize(watermark_image.width, watermark_image.height)
    mask = watermark_image.split()[3].point(lambda i: i * cfg.watermark_opacity)
    pos = (
        (im.width - watermark_image.width - 25),
        (im.height - watermark_image.height - 25),
    )
    im.paste(watermark_image, pos, mask)
    return im


def with_text(im: Image, cfg: Config, ctx: Context) -> Image:
    """Watermark with text if program option is supplied.

    If text is equal to 'copyright', the exif data will be read
    (if available) to attempt to determine copyright date based on
    date photo was taken.

    Parameters
    ----------
    - `im` PIL Image
    - `cfg` Config object
    - `ctx` Context object

    """
    if cfg.text is None and cfg.text_copyright is None:
        LOG.error("Missing text or copyright text in cfg")
        return im
    if cfg.text_copyright is not None:
        photo_dt = datetime.now()
        if ctx.orig_exif is not None:
            try:
                photo_dt = datetime.strptime(
                    ctx.orig_exif["Exif"]["DateTimeOriginal"].decode("utf-8"),
                    "%Y:%m:%d %H:%M:%S",
                )
            except KeyError:
                pass
        copyright_year = photo_dt.strftime("%Y")
        cfg.text = f"© {copyright_year} {cfg.text_copyright}"
        LOG.info("Using copyright text: %s", cfg.text)
        LOG.info("Photo date from exif: %s", photo_dt)
    layer = Image.new("RGBA", (im.width, im.height), (255, 255, 255, 0))

    font_size = 1  # starting size
    offset_x = 10
    offset_y = 10

    try:
        cwd = PurePath(os.path.dirname(__file__))
        font_path = str(PurePath.joinpath(cwd, "fonts", "SourceSansPro-Regular.ttf"))
        font = ImageFont.truetype(font=font_path, size=font_size)
    except OSError:
        LOG.error("Could not find font '%s', aborting text watermark", font_path)
        return im

    LOG.debug("Found font '%s'", font_path)
    while font.getsize(cfg.text)[0] < cfg.text_scale * im.width:
        # iterate until text size is >= text_scale
        font_size += 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    if font.getsize(cfg.text)[0] > cfg.text_scale * im.width:
        font_size -= 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    text_width, text_height = font.getsize(cfg.text)
    stats = ImageStat.Stat(
        im.crop((offset_x, offset_y, text_width, im.height - text_height)).convert("L")
    )
    LOG.debug(
        "Final text dims: %d x %d px; Font size: %d", text_width, text_height, font_size
    )
    LOG.debug("Text region stats: mean=%f, rms=%f", stats.mean[0], stats.rms[0])
    d = ImageDraw.Draw(layer)
    opacity = int(round((cfg.text_opacity * 255)))
    LOG.info("Text opacity: %d/255", opacity)

    text_fill = 255, 255, 255, opacity
    if stats.mean[0] > 60:
        text_fill = 0, 0, 0, opacity

    d.text((offset_x, offset_y), cfg.text, font=font, fill=text_fill)
    out = Image.alpha_composite(im.convert("RGBA"), layer)
    return out.convert("RGB")
