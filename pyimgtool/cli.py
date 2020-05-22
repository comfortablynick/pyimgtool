"""Resize and watermark images."""

import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from pprint import pformat
from time import perf_counter
from typing import Dict, Optional

import cv2
import numpy as np
import piexif
import plotille
from PIL import Image
from sty import ef, fg, rs

from pyimgtool.args import parse_args
from pyimgtool.commands import resize, watermark
from pyimgtool.data_structures import ImageSize
from pyimgtool.utils import humanize_bytes, escape_ansi

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger(__name__)


def main():
    """Process image based on cli args."""
    time_start = perf_counter()

    args = parse_args(sys.argv[1:]).ordered()
    _, opts = next(args)
    log_level = 0
    try:
        log_level = (0, 20, 10)[opts.verbosity]
    except IndexError:
        log_level = 10
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # set level for all loggers
    for l in loggers:
        l.setLevel(log_level)

    LOG.debug("Program opts:\n%s", pformat(vars(opts)))

    # main vars
    im: Image = None
    in_file_path: str = None
    in_image_size = ImageSize(0, 0)
    in_file_size = 0
    in_dpi = 0
    in_exif: Optional[Dict] = None
    out_file_path = None
    out_image_size = ImageSize(0, 0)
    out_file_size = 0
    no_op = False

    for cmd, arg in args:
        LOG.debug("Processing command %s with args:\n%s", cmd, pformat(vars(arg)))

        if cmd == "open":
            inbuf = BytesIO()
            inbuf.write(arg.input.read())
            in_file_size = inbuf.tell()
            im = Image.open(inbuf)
            assert im is not None

            in_image_size = ImageSize(*im.size)
            LOG.info("Input dims: %s", in_image_size)
            in_file_path = arg.input.name
            try:
                in_exif = piexif.load(in_file_path)
                # del in_exif["thumbnail"]
            except KeyError:
                pass
            in_dpi = im.info["dpi"]
            LOG.info("Input size: %s", humanize_bytes(in_file_size))
            if arg.show_histogram:
                print(generate_rgb_histogram(im))
        elif cmd == "resize":
            new_size = resize.calculate_new_size(
                in_image_size, arg.scale, ImageSize(width=arg.width, height=arg.height),
            )
            out_image_size = ImageSize(width=new_size.width, height=new_size.height)

            # Resize/resample
            im = resize.resize_thumbnail(
                im,
                out_image_size,
                #  bg_size=(cfg.width + 50, cfg.height + 50),
                resample=Image.ANTIALIAS,
            )
        elif cmd == "text":
            im = watermark.with_text(
                im,
                text=arg.text,
                copyright=arg.copyright,
                scale=arg.scale,
                position=arg.position,
                opacity=arg.opacity,
                exif=in_exif,
            )
        elif cmd == "watermark":
            im = watermark.with_image(
                im,
                Image.open(arg.image),
                scale=arg.scale,
                position=arg.position,
                opacity=arg.opacity,
            )
        elif cmd == "save":
            use_progressive_jpg = in_file_size > 10000
            if use_progressive_jpg:
                LOG.debug("Large file; using progressive jpg")

            # Exif
            if arg.keep_exif:
                exif = piexif.dump(piexif.load(in_file_path))
            else:
                exif = b""

            outbuf = BytesIO()
            im.save(
                outbuf,
                "JPEG",
                quality=arg.jpg_quality,
                dpi=in_dpi,
                progressive=use_progressive_jpg,
                optimize=True,
                exif=exif,
            )
            image_buffer = outbuf.getbuffer()
            out_file_size = image_buffer.nbytes
            LOG.info("Output size: %s", humanize_bytes(out_file_size))

            if arg.output is not None:
                out_file_path = arg.output.name
                LOG.info("Saving buffer to %s", arg.output.name)

            if arg.no_op:
                no_op = True
                continue
            if (out_path := Path(out_file_path)).exists():
                if not arg.force:
                    LOG.critical(
                        "file '%s' exists and force argument not found", out_path
                    )
                    print(
                        f"{fg.red}{ef.bold}Error: file '{out_path}' exists;",
                        f" use -f option to force overwrite.{rs.all}",
                        file=sys.stderr,
                    )
                    return
                # Create output dir if it doesn't exist
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with out_path.open("wb") as f:
                    f.write(image_buffer)

    time_end = perf_counter()
    size_reduction_bytes = in_file_size - out_file_size
    no_op_msg = f" **Image not saved due to -n flag; reporting only** "
    # no_op_msg = " **Image not saved due to -n flag; reporting only** "
    report_title = " Processing Summary "
    report_end = " End "
    report_arrow = "->"
    report = []
    report.append(
        [
            "File Name:",
            in_file_path,
            report_arrow if out_file_path is not None else "",
            out_file_path if out_file_path is not None else "",
        ]
    )
    report.append(
        ["File Dimensions:", str(in_image_size), report_arrow, str(out_image_size)]
    )
    report.append(
        [
            "File Size:",
            humanize_bytes(in_file_size),
            report_arrow,
            humanize_bytes(out_file_size),
        ]
    )
    report.append(
        [
            "Size Reduction:",
            f"{humanize_bytes(size_reduction_bytes)} "
            f"({(size_reduction_bytes/in_file_size) * 100:2.1f}%)",
        ]
    )
    report.append(["Processing Time:", f"{(time_end - time_start)*1000:.1f} ms"])
    for c in report:
        for n in range(4):
            try:
                c[n] = c[n]
            except IndexError:
                c.append("")
        c[2] = "" if c[3] == c[1] else c[2]
        c[3] = "  " if c[3] == c[1] else c[3]

    padding = 2
    col0w = max([len(str(c[0])) for c in report]) + padding
    col1w = max([len(str(c[1])) for c in report]) + padding
    col2w = max([len(str(c[2])) for c in report]) + padding
    col3w = max([len(str(c[3])) for c in report]) + padding
    out = []
    out.append(f"{ef.b}{report_title:{'-'}^{col0w + col1w + col2w + col3w + 1}}{rs.all}")
    if no_op:
        out.append(f"{fg.li_cyan}{ef.b}{no_op_msg:^{col0w + col1w + col2w + col3w + 1}}{rs.all}")
    for line in report:
        out.append(
            f"{line[0]:<{col0w}}{rs.all} {line[1]:{col1w}} {line[2]:{col2w}} {ef.i}{line[3]:{col3w}}{rs.all}"
        )
    out.append(f"{ef.b}{report_end:{'-'}^{col0w + col1w + col2w + col3w + 1}}{rs.all}")
    print(*out, sep="\n")


def generate_rgb_histogram(im: Image, show_axes: bool = False) -> str:
    """Return string of histogram for image to print in terminal.

    Args:
        im: PIL Image object
        show_axes: Print x and y axes

    Returns: String of histscalee
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
