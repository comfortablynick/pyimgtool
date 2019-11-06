"""Image operations."""

import logging
import sys
from io import BytesIO

import cv2
import numpy as np
import piexif
import plotille
from PIL import Image

from pyimgtool import resize, watermark
from pyimgtool.data_structures import Config, Context, ImageSize
from pyimgtool.utils import humanize_bytes

LOG = logging.getLogger(__name__)


def generate_histogram(cfg: Config) -> str:
    """Return string of histogram for image to print in terminal.

    Parameters
    ----------
    - `cfg` Config object

    Returns
    -------
    - Str of histogram content

    """
    hist_bins = 256
    img = Image.open(cfg.input_file).convert("L")
    # resize to make histogram faster
    new_size = calculate_new_size(0.0, ImageSize(*img.size), ImageSize(1000, 0))
    flat = cv2.resize(np.asarray(img), dsize=(new_size.width, new_size.height)).ravel()
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


# TODO: improve this func to be more comprehensive
def calculate_new_size(
    pct_scale: float, orig_size: ImageSize, new_size: ImageSize
) -> ImageSize:
    """Calculate new dimensions and maintain image aspect ratio.

    Pct scale is given precedence over new size dims.

    Parameters
    ----------
    - `pct_scale` Scale by this factor
    - `orig_size` ImageSize object of original file dims
    - `new_size` ImageSize object of desired new dims

    Returns
    -------
    - ImageSize object of correct proprotions for new size

    """
    if pct_scale is not None and pct_scale > 0.0:
        LOG.info("Scaling image by %.1f%%", pct_scale)
        new_size.width = int(round(orig_size.width * (pct_scale / 100.0)))
        new_size.height = int(round(orig_size.height * (pct_scale / 100.0)))
    elif new_size.width > 0 and new_size.height <= 0:
        LOG.info("Calculating height based on width")
        new_size.height = int(
            round((new_size.width * orig_size.height) / orig_size.width)
        )
    elif new_size.height > 0 and new_size.width <= 0:
        LOG.info("Calculating width based on height")
        new_size.width = int(
            round((new_size.height * orig_size.width) / orig_size.height)
        )
    else:
        LOG.info("No new width, height, or pct scale supplied; using current dims")
        new_size.width = orig_size.width
        new_size.height = orig_size.height
    LOG.debug("New size: %s", new_size)
    return new_size


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
        cfg.pct_scale, ctx.orig_size, ImageSize(width=cfg.width, height=cfg.height)
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
            print(generate_histogram(cfg))
        ctx.new_size.width, ctx.new_size.height = img_out.size
        ctx.new_file_size = sys.getsizeof(ctx.image_buffer)
    LOG.info("Output size: %s", humanize_bytes(ctx.new_file_size))
    return ctx
