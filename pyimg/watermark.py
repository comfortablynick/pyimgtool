"""Watermark image with text or another image."""

import logging
import os

from PIL import Image

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
