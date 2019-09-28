"""Watermark image with text or another image."""

import logging
import os
from datetime import datetime
from string import Template

from PIL import Image, ImageDraw, ImageFont

from pyimg import resize
from pyimg.data_structures import Config, Context, ImageSize

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
    font_path = "DejaVuSans.ttf"
    font = ImageFont.truetype(font=font_path, size=font_size)

    while font.getsize(cfg.text)[0] < cfg.text_scale * im.width:
        # iterate until text size is >= text_scale
        font_size += 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    if font.getsize(cfg.text)[0] > cfg.text_scale * im.width:
        font_size -= 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    LOG.debug("Final text dims: %s; Font size: %d", font.getsize(cfg.text), font_size)
    d = ImageDraw.Draw(layer)
    opacity = int(round((cfg.text_opacity * 255)))
    LOG.info("Text opacity: %d/255", opacity)
    d.text((10, 10), cfg.text, font=font, fill=(255, 255, 255, opacity))
    out = Image.alpha_composite(im.convert("RGBA"), layer)
    return out.convert("RGB")
