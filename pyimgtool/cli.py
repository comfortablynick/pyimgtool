"""Resize and watermark images."""

import logging
import sys
from pathlib import Path
from time import perf_counter

from sty import ef, fg, rs

from pyimgtool.args import parse_args
from pyimgtool.data_structures import Config
from pyimgtool.image import process_image
from pyimgtool.utils import get_summary_report

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
