"""Image operations."""

import logging
import sys
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import piexif
import plotille
from PIL import Image

from pyimgtool import resize, watermark
from pyimgtool.data_structures import Config, Context, ImageSize
from pyimgtool.utils import humanize_bytes

LOG = logging.getLogger(__name__)


def generate_histogram(im: Image, cfg: Config) -> str:
    """Return string of histogram for image to print in terminal.

    Parameters
    ----------
    - `cfg` Config object

    Returns
    -------
    - Str of histogram content

    """
    hist_bins = 256
    img = np.asarray(im.convert("L"))
    # resize to 10% of orig to make histogram faster
    new_size = calculate_new_size(ImageSize(*img.shape[:2]), 10.0, None)
    LOG.debug("Img resized to %s for histogram", new_size)
    flat = cv2.resize(np.asarray(img), dsize=(new_size.width, new_size.height))
    hist_data, bins = np.histogram(flat, bins=range(hist_bins + 1), range=[0, 256])
    hist = plotille.plot(
        bins[:hist_bins],
        hist_data,
        width=50,
        height=10,
        x_min=0,
        x_max=255,
        y_min=0,
        y_max=float(hist_data.max()),
    )
    return str(hist)


def calculate_new_size(
    orig_size: ImageSize, pct_scale: Optional[float], new_size: Optional[ImageSize]
) -> ImageSize:
    """Calculate new dimensions and maintain image aspect ratio.

    Pct scale is given precedence over new size dims.

    Parameters
    ----------
    - `orig_size` ImageSize object of original file dims
    - `pct_scale` Optional factor to scale by (1.0-100.0)
    - `new_size` Optional ImageSize object of desired new dims

    Returns
    -------
    - ImageSize object of correct proprotions for new size

    """
    calc_size = ImageSize()
    if pct_scale is not None and pct_scale > 0.0:
        LOG.info("Scaling image by %.1f%%", pct_scale)
        calc_size.width = int(round(orig_size.width * (pct_scale / 100.0)))
        calc_size.height = int(round(orig_size.height * (pct_scale / 100.0)))
        return calc_size

    if new_size is not None and new_size != calc_size:
        if new_size.width > 0 and new_size.height <= 0:
            LOG.info("Calculating height based on width")
            calc_size.width = new_size.width
            calc_size.height = int(
                round((calc_size.width * orig_size.height) / orig_size.width)
            )
        elif new_size.height > 0 and new_size.width <= 0:
            LOG.info("Calculating width based on height")
            calc_size.height = new_size.height
            calc_size.width = int(
                round((calc_size.height * orig_size.width) / orig_size.height)
            )
        return calc_size
    LOG.info("No new width, height, or pct scale supplied; using current dims")
    return orig_size


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
        exif = piexif.load(cfg.input_file)
        del exif["thumbnail"]
        ctx.orig_exif = exif
    except KeyError:
        pass
    ctx.orig_size.width, ctx.orig_size.height = im.size
    ctx.orig_dpi = im.info["dpi"]
    LOG.info("Input dims: %s", ctx.orig_size)
    LOG.info("Input size: %s", humanize_bytes(ctx.orig_file_size))

    new_size = calculate_new_size(
        ctx.orig_size, cfg.pct_scale, ImageSize(width=cfg.width, height=cfg.height)
    )
    cfg.width = new_size.width
    cfg.height = new_size.height

    # Resize/resample
    if cfg.height != ctx.orig_size.height or cfg.width != ctx.orig_size.width:
        im = resize.resize_thumbnail(
            im,
            (cfg.width, cfg.height),
            #  bg_size=(cfg.width + 50, cfg.height + 50),
            resample=Image.ANTIALIAS,
        )

    if cfg.watermark_image is not None:
        im = watermark.with_image(im, cfg, ctx)
    if cfg.text is not None or cfg.text_copyright is not None:
        im = watermark.with_text(im, cfg, ctx)

    try:
        ctx.new_dpi = im.info["dpi"]
    except KeyError:
        pass
    LOG.info("Image mode: %s", im.mode)

    # Save
    use_progressive_jpg = ctx.orig_file_size > 10000
    if use_progressive_jpg:
        LOG.debug("Large file; using progressive jpg")

    # Exif
    if cfg.keep_exif:
        exif = piexif.dump(piexif.load(cfg.input_file))
    else:
        exif = b""

    im.save(
        outbuf,
        "JPEG",
        quality=cfg.jpg_quality,
        dpi=ctx.orig_dpi,
        progressive=use_progressive_jpg,
        optimize=True,
        exif=exif,
    )
    ctx.image_buffer = outbuf.getvalue()

    # convert back to image to get size
    if ctx.image_buffer:
        img_out = Image.open(BytesIO(ctx.image_buffer))
        if cfg.show_histogram:
            # print(generate_histogram(cfg))
            print(generate_histogram(im, cfg))
        ctx.new_size.width, ctx.new_size.height = img_out.size
        ctx.new_file_size = sys.getsizeof(ctx.image_buffer)
    LOG.info("Output size: %s", humanize_bytes(ctx.new_file_size))
    return ctx
