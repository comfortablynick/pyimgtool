"""Image operations."""

import logging
import sys
from io import BytesIO

import piexif
from PIL import Image

from pyimg import resize, watermark
from pyimg.data_structures import Config, ImageContext

LOG = logging.getLogger(__name__)


def humanize_bytes(num, suffix="B", si_prefix=False, round_digits=2) -> str:
    """Return a human friendly byte representation.

    Modified from: https://stackoverflow.com/questions/1094841/1094933#1094933
    """
    div = 1000.0 if si_prefix else 1024.0
    unit_suffix = "i" if si_prefix else ""
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < div:
            return f"{num:3.{round_digits}f} {unit}{unit_suffix}{suffix}"
        num /= div
    return f"{num:3.{round_digits}f} Y{unit_suffix}{suffix}"


def calculate_new_size(cfg: Config, ctx: ImageContext):
    """Update Config with correct width/height.

    Percent scale (-p) takes precedence over width (-mw) and height (-mh).
    """
    if cfg.pct_scale:
        LOG.info("Scaling image by %.1f%%", cfg.pct_scale)
        cfg.width = int(round(ctx.orig_size.width * (cfg.pct_scale / 100.0)))
        cfg.height = int(round(ctx.orig_size.height * (cfg.pct_scale / 100.0)))

    if cfg.width and not cfg.height:
        LOG.info("Calculating height based on width")
        cfg.height = int(
            round((cfg.width * ctx.orig_size.height) / ctx.orig_size.width)
        )
    elif cfg.height and not cfg.width:
        LOG.info("Calculating width based on height")
        cfg.width = int(
            round((cfg.height * ctx.orig_size.width) / ctx.orig_size.height)
        )
    else:
        LOG.info("No new width or height supplied; using current dims")
        cfg.width = ctx.orig_size.width
        cfg.height = ctx.orig_size.height


def process_image(cfg: Config) -> ImageContext:
    """Process image according to options in `cfg`."""
    ctx = ImageContext()
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
    elif cfg.watermark_text is not None:
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
    im.save(outbuf, "JPEG", quality=cfg.jpg_quality, dpi=ctx.orig_dpi)
    ctx.image_buffer = outbuf.getvalue()

    # convert back to image to get size
    if ctx.image_buffer:
        img_out = Image.open(BytesIO(ctx.image_buffer))
        ctx.new_size = img_out.size
        ctx.new_file_size = sys.getsizeof(ctx.image_buffer)
    LOG.info("Output size: %s", humanize_bytes(ctx.new_file_size))
    return ctx
