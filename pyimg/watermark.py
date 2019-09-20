"""Watermark image with text or another image."""

import logging
import os

from PIL import Image, ImageDraw, ImageFont

from pyimg.data_structures import Config, ImageContext

LOG = logging.getLogger(__name__)


def with_image(im: Image, cfg: Config, ctx: ImageContext) -> Image:
    """Watermark with image according to Config."""
    if cfg.watermark_image is None:
        LOG.error("Missing watermark_image in cfg")
        return im
    watermark_image = Image.open(os.path.expanduser(cfg.watermark_image)).convert(
        "RGBA"
    )
    mask = watermark_image.split()[3].point(lambda i: i * cfg.watermark_opacity)
    pos = (
        (ctx.orig_size.width - watermark_image.width - 25),
        (ctx.orig_size.height - watermark_image.height - 25),
    )
    im.paste(watermark_image, pos, mask)
    return im


def with_text(im: Image, cfg: Config, ctx: ImageContext) -> Image:
    """Watermark with text according to Config."""
    if cfg.watermark_text is None:
        LOG.error("Missing watermark_text in cfg")
        return im
    layer = Image.new(
        "RGBA", (ctx.orig_size.width, ctx.orig_size.height), (255, 255, 255, 0)
    )
    font = ImageFont.truetype("SourceSansPro-Regular.ttf", 16)
    d = ImageDraw.Draw(layer)
    d.text((10, 10), cfg.watermark_text, font=font, fill=(255, 255, 255, 128)) # last num is alpha
    out = Image.alpha_composite(im, layer)
    return out
