"""Resize and watermark images."""

import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
import piexif
import plotille
from PIL import Image
from sty import ef, fg, rs

from pyimgtool.args import parse_args
from pyimgtool.commands import resize, watermark
from pyimgtool.data_structures import Config, Context, ImageSize
from pyimgtool.utils import get_summary_report, humanize_bytes

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger(__name__)


def main():
    """Process image based on config file and command-line arguments."""
    time_start = perf_counter()
    cfg = Config.from_args(parse_args(sys.argv[1:]))
    log_level = 0
    try:
        log_level = (0, 20, 10)[cfg.verbosity]
    except IndexError:
        log_level = 10
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # set level for all loggers
    for l in loggers:
        l.setLevel(log_level)

    LOG.debug("Runtime config:\n%s", cfg)
    ctx = process_image(cfg)
    ctx.time_start = time_start
    exclude_ctx_attrs = ["image_buffer"]
    if cfg.verbosity < 3:
        exclude_ctx_attrs.append("orig_exif")
    LOG.debug(
        "Image Context:\n%s", ctx,
    )

    if not cfg.no_op:
        if not ctx.image_buffer:
            LOG.critical("Image buffer cannot be None")
            raise ValueError("Image buffer is None")
        LOG.info("Saving buffer to %s", cfg.output_file)

        # Create output dir if it doesn't exist
        out_path = Path(cfg.output_file)
        if out_path.exists():
            # output file exists
            if not cfg.force:
                print(
                    fg.red
                    + ef.bold
                    + f"Error: file '{out_path}' exists; use -f option to force overwrite."
                    + rs.all,
                    file=sys.stderr,
                )
                return
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("wb") as f:
            f.write(ctx.image_buffer)
    else:
        print(fg.li_magenta + "***Displaying Results Only***" + fg.rs)

    ctx.time_end = perf_counter()
    print(*get_summary_report(cfg, ctx), sep="\n")


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

    new_size = resize.calculate_new_size(
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
            print(generate_rgb_histogram(im))
        ctx.new_size.width, ctx.new_size.height = img_out.size
        ctx.new_file_size = sys.getsizeof(ctx.image_buffer)
    LOG.info("Output size: %s", humanize_bytes(ctx.new_file_size))
    return ctx


def generate_rgb_histogram(im: Image, show_axes: bool = False) -> str:
    """Return string of histogram for image to print in terminal.

    Args:
        im: PIL Image object
        show_axes: Print x and y axes

    Returns: String of histogram content for rgb image
    """
    hist_width = 50
    hist_height = 10
    hist_bins = 256

    # set up graph
    fig = plotille.Figure()
    fig.width = hist_width
    fig.height = hist_height
    fig.origin = False  # Don't draw 0 lines
    fig.set_x_limits(min_=0, max_=hist_bins - 1)
    fig.set_y_limits(min_=0)
    fig.color_mode = "names"

    img = np.asarray(im)
    img_h, img_w, img_c = img.shape
    colors = ["red", "green", "blue"]

    for i in range(img_c):
        hist_data, bins = np.histogram(
            img[..., i], bins=range(hist_bins + 1), range=[0, 256]
        )
        fig.plot(bins[:hist_bins], hist_data, lc=colors[i])
    if not show_axes:
        graph = (
            "\n".join(["".join(l.split("|")[1]) for l in fig.show().splitlines()[1:-2]])
            + "\n"
        )
    else:
        graph = fig.show()
    return graph
