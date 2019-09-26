"""Image operations."""

import logging
import sys
from io import BytesIO
from typing import Optional, Tuple

import attr
import piexif
from PIL import Image

from pyimg import resize, watermark
from pyimg.data_structures import Config, Context
from pyimg.utils import humanize_bytes

LOG = logging.getLogger(__name__)


def calculate_new_size(cfg: Config, ctx: Context):
    """Update Config with correct width/height.

    Percent scale (`-p`) takes precedence over width (`-mw`) and
    height (`-mh`). Func does nothing if both height and width
    are supplied at the command line.

    """
    if cfg.pct_scale:
        LOG.info("Scaling image by %.1f%%", cfg.pct_scale)
        cfg.width = int(round(ctx.orig_size.width * (cfg.pct_scale / 100.0)))
        cfg.height = int(round(ctx.orig_size.height * (cfg.pct_scale / 100.0)))
    elif cfg.width and not cfg.height:
        LOG.info("Calculating height based on width")
        cfg.height = int(
            round((cfg.width * ctx.orig_size.height) / ctx.orig_size.width)
        )
    elif cfg.height and not cfg.width:
        LOG.info("Calculating width based on height")
        cfg.width = int(
            round((cfg.height * ctx.orig_size.width) / ctx.orig_size.height)
        )
    elif not cfg.height and not cfg.width:
        LOG.info("No new width or height supplied; using current dims")
        cfg.width = ctx.orig_size.width
        cfg.height = ctx.orig_size.height


def process_image(cfg: Config) -> Context:
    """Process image according to options in `cfg`."""
    ctx = Context()
    inbuf = BytesIO()
    outbuf = BytesIO()
    if not cfg.input_file:
        raise ValueError("input_file required")
    with open(cfg.input_file, "rb") as f:
        inbuf.write(f.read())
    ctx.orig_file_size = inbuf.tell()
    im = Image.open(inbuf)
    try:
        exif = piexif.load(im.info["exif"], True)
        del exif["thumbnail"]
        ctx.orig_exif = exif
    except KeyError:
        pass
    ctx.orig_size.width, ctx.orig_size.height = im.size
    ctx.orig_dpi = im.info["dpi"]
    LOG.info("Input dims: %s", ctx.orig_size)
    LOG.info("Input size: %s", humanize_bytes(ctx.orig_file_size))

    calculate_new_size(cfg, ctx)

    if cfg.watermark_image is not None:
        im = watermark.with_image(im, cfg, ctx)
    if cfg.text is not None:
        im = watermark.with_text(im, cfg, ctx)

    # Resize/resample
    if cfg.height != ctx.orig_size.height or cfg.width != ctx.orig_size.width:
        im = resize.resize_thumbnail(
            im,
            (cfg.width, cfg.height),
            #  bg_size=(cfg.width + 50, cfg.height + 50),
            resample=Image.ANTIALIAS,
        )

    try:
        ctx.new_dpi = im.info["dpi"]
    except KeyError:
        pass
    LOG.info("Image mode: %s", im.mode)

    # Save
    im.save(outbuf, "JPEG", quality=cfg.jpg_quality, dpi=ctx.orig_dpi, progressive=True)
    ctx.image_buffer = outbuf.getvalue()

    # convert back to image to get size
    if ctx.image_buffer:
        img_out = Image.open(BytesIO(ctx.image_buffer))
        ctx.new_size.width, ctx.new_size.height = img_out.size
        ctx.new_file_size = sys.getsizeof(ctx.image_buffer)
    LOG.info("Output size: %s", humanize_bytes(ctx.new_file_size))
    return ctx
