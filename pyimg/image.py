"""Image operations."""

import logging
import os
import sys
from io import BytesIO

import tinify
from PIL import Image

from pyimg.data_structures import Config

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


def process_image(cfg: Config) -> bytes:
    """Process image according to options in `cfg`."""
    inbuf = BytesIO()
    outbuf = BytesIO()
    if not cfg.input_file:
        raise ValueError("input_file required")
    with open(cfg.input_file, "rb") as f:
        inbuf.write(f.read())
    orig_size = inbuf.tell()
    im = Image.open(inbuf)
    in_width, in_height = im.size
    LOG.info("Input dims: %s", (in_width, in_height))
    LOG.info("Input size: %s", humanize_bytes(orig_size))

    # get new dims from args
    if cfg.pct_scale:
        LOG.info("Scaling image by %.1f%%", cfg.pct_scale)
        cfg.width = int(round(in_width * (cfg.pct_scale / 100.0)))
        cfg.height = int(round(in_height * (cfg.pct_scale / 100.0)))
    if cfg.width and not cfg.height:
        LOG.info("Calculating height based on width")
        cfg.height = int(round((cfg.width * in_height) / in_width))
    elif cfg.height and not cfg.width:
        LOG.info("Calculating width based on height")
        cfg.width = int(round((cfg.height * in_width) / in_height))

    if cfg.watermark_image:
        watermark_image = Image.open(os.path.expanduser(cfg.watermark_image)).convert(
            "RGBA"
        )

        mask = watermark_image.split()[3].point(lambda i: i * cfg.watermark_opacity)
        pos = (
            (in_width - watermark_image.width - 25),
            (in_height - watermark_image.height - 25),
        )
        im.paste(watermark_image, pos, mask)

    if cfg.use_tinify:
        im.save(outbuf, "JPEG")
        try:
            outbuf = (
                tinify.tinify.from_buffer(outbuf.getvalue())
                .resize(method="fit", width=cfg.width, height=cfg.height)
                .to_buffer()
            )
            LOG.info("Tinify monthly count: %d", tinify.tinify.compression_count)
        except tinify.errors.AccountError:
            print(
                "Tinify API key invalid; check API key and try again.", file=sys.stderr
            )
            LOG.critical(
                "Aborting due to invalid Tinify API key: '%s'", cfg.tinify_api_key
            )
            sys.exit(1)
    else:
        im.thumbnail((cfg.width, cfg.height), Image.ANTIALIAS)
        out_width, out_height = im.size
        LOG.info("Output dims: %s", (out_width, out_height))
        im.save(outbuf, "JPEG", quality=cfg.jpg_quality)
        out = outbuf.getvalue()
    new_size = sys.getsizeof(out)
    LOG.info("Output size: %s", humanize_bytes(new_size))
    return out
