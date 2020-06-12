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
from pyimgtool.commands import mat, resize, watermark, sharpen
from pyimgtool.data_structures import Size
from pyimgtool.exceptions import (
    ImageTooSmallError,
    ResizeAttributeError,
    ResizeNotNeededError,
)
from pyimgtool.utils import humanize_bytes

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
    for logger in loggers:
        logger.setLevel(log_level)

    LOG.debug("Program opts:\n%s", pformat(vars(opts)))

    # main vars
    im: Image = None
    in_file_path: str = None
    in_image_size = Size(0, 0)
    in_file_size = 0
    in_dpi = 0
    in_exif: Optional[Dict] = None
    out_exif: bytes = b""
    out_exif_size = 0
    out_file_path = None
    out_image_size = Size(0, 0)
    out_file_size = 0
    no_op = False

    for cmd, arg in args:
        LOG.debug("Processing command %s with args:\n%s", cmd, pformat(vars(arg)))

        if cmd == "open":
            in_file_path = arg.input.name
            in_file_size = os.path.getsize(in_file_path)
            im = Image.open(arg.input)
            in_image_size = Size(*im.size)
            LOG.info("Input dims: %s", in_image_size)
            try:
                in_exif = piexif.load(in_file_path)
                del in_exif["thumbnail"]
                # LOG.debug("Exif: %s", in_exif)
                in_dpi = im.info["dpi"]
            except KeyError:
                pass
            LOG.info("Input file size: %s", humanize_bytes(in_file_size))
            LOG.info("Input dpi: %s", in_dpi)
            if arg.show_histogram:
                print(generate_rgb_histogram(im))
        elif cmd == "mat":
            im = np.asarray(im) if type(im) != np.ndarray else im
            im = mat.create_mat(im, size_inches=arg.size)
            out_image_size = Size.from_np(im)
        elif cmd == "resize":
            orig_size = Size(*im.size)
            out_image_size = orig_size
            try:
                resize_method, new_size = resize.get_method(
                    orig_size,
                    width=arg.width,
                    height=arg.height,
                    scale=arg.scale,
                    longest=arg.longest,
                    shortest=arg.shortest,
                )
            except ResizeNotNeededError as e:
                LOG.warning(e)
            else:
                # Resize/resample
                try:
                    im = resize.resize(resize_method, im, new_size,)
                except ImageTooSmallError as e:
                    LOG.warning(e)
                out_image_size = Size(*im.size)
        elif cmd == "resize2":
            im = np.asarray(im)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            orig_size = Size.from_np(im)
            out_image_size = orig_size
            y, x = im.shape[:2]
            try:
                resize_method, new_size = resize.get_method(
                    orig_size,
                    width=arg.width,
                    height=arg.height,
                    scale=arg.scale,
                    longest=arg.longest,
                    shortest=arg.shortest,
                    force=arg.force,
                )
            except ResizeNotNeededError as e:
                LOG.warning(e)
            except ResizeAttributeError as e:
                print(f"{fg.li_red}error: {e}{rs.fg}", file=sys.stderr)
                sys.exit(1)
            else:
                try:
                    im = resize.resize_opencv(
                        resize_method, im, new_size, resample=cv2.INTER_LANCZOS4
                    )
                except ImageTooSmallError as e:
                    LOG.warning(e)
                out_image_size = Size.from_np(im)
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
        elif cmd == "text2":
            im = watermark.with_text(
                Image.fromarray(im),
                text=arg.text,
                copyright=arg.copyright,
                scale=arg.scale,
                position=arg.position,
                opacity=arg.opacity,
                exif=in_exif,
            )
            im = np.asarray(im)
        elif cmd == "watermark":
            im = watermark.with_image(
                im,
                Image.open(arg.image),
                scale=arg.scale,
                position=arg.position,
                opacity=arg.opacity,
            )
        elif cmd == "watermark2":
            watermark_image = cv2.imread(arg.image.name, cv2.IMREAD_UNCHANGED)
            im = watermark.with_image_opencv(
                im,
                watermark_image,
                scale=arg.scale,
                position=arg.position,
                opacity=arg.opacity,
            )
        elif cmd == "sharpen":
            im = sharpen.unsharp_mask(im, amount=arg.amount, threshold=arg.threshold)
        elif cmd == "save":
            use_progressive_jpg = in_file_size > 10000
            if use_progressive_jpg:
                LOG.debug("Large file; using pSizee jpg")

            # Exif
            if arg.keep_exif:
                out_exif = piexif.dump(piexif.load(in_file_path))
                out_exif_size = sys.getsizeof(out_exif)

            outbuf = BytesIO()
            try:
                im.save(
                    outbuf,
                    "JPEG",
                    quality=arg.jpg_quality,
                    dpi=in_dpi,
                    progressive=use_progressive_jpg,
                    optimize=True,
                    exif=out_exif,
                )
            except AttributeError:
                write_params = [
                    cv2.IMWRITE_JPEG_QUALITY,
                    arg.jpg_quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                ]
                if use_progressive_jpg:
                    write_params += [
                        cv2.IMWRITE_JPEG_PROGRESSIVE,
                    ]
                _, buf = cv2.imencode(".jpg", im, write_params)
                outbuf = BytesIO(buf)
            image_buffer = outbuf.getbuffer()
            out_file_size = image_buffer.nbytes + out_exif_size
            LOG.info("Buffer output size: %s", humanize_bytes(out_file_size))

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
                    if arg.keep_exif:
                        piexif.insert(out_exif, out_file_path)
                    out_file_size = os.path.getsize(out_file_path)

    time_end = perf_counter()
    elapsed = time_end - time_start
    report = generate_report(
        in_file_size,
        out_file_size,
        in_file_path,
        out_file_path,
        in_image_size,
        out_image_size,
        elapsed,
        no_op,
    )
    print(report)


def generate_report(
    in_file_size: int,
    out_file_size: int,
    in_file_path: str,
    out_file_path: str,
    in_image_size: Size,
    out_image_size: Size,
    elapsed_time: float,
    no_op: bool = False,
) -> str:
    """
    Generate report to display on terminal after processing.

    Parameters
    ----------
    in_file_size : int
        File size, in bytes, of original file.
    out_file_size : int
        File size, in bytes, of edited file.
    in_file_path : str
        Original file path.
    out_file_path : str
        New file path.
    in_image_size : Size
        Original image size.
    out_image_size : Size
        Edited image size.
    elapsed_time : float
        Elapsed time from perf_counter() of editing operations.
    no_op : bool
        Dry run; no image is being saved.

    Returns
    -------
    str:
        Report text
    """
    size_reduction_bytes = in_file_size - out_file_size
    no_op_msg = "**Image not saved due to -n flag; reporting only**"
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
    report.append(["Processing Time:", f"{elapsed_time*1000:.1f} ms"])
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
    out.append(
        f"{ef.b}{report_title:{'-'}^{col0w + col1w + col2w + col3w + 1}}{rs.all}"
    )
    if no_op:
        out.append(
            f"{fg.li_cyan}{ef.b}{no_op_msg:^{col0w + col1w + col2w + col3w + 1}}{rs.all}"
        )
    for line in report:
        out.append(
            f"{line[0]:<{col0w}}{rs.all} {line[1]:{col1w}}"
            + f"{line[2]:{col2w}} {ef.i}{line[3]:{col3w}}{rs.all}"
        )
    out.append(f"{ef.b}{report_end:{'-'}^{col0w + col1w + col2w + col3w + 1}}{rs.all}")
    return "\n".join(out)


def generate_rgb_histogram(im: Image, show_axes: bool = False) -> str:
    """Generate histogram for terminal.

    Parameters
    ----------
    im
        Image to evaluate
    show_axes
        Show x and y axis labels

    Returns
    -------
    str:
        Histogram to print
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
            "\n".join(
                ["".join(ln.split("|")[1]) for ln in fig.show().splitlines()[1:-2]]
            )
            + "\n"
        )
    else:
        graph = fig.show()
    return graph
