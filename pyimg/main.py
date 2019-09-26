"""Resize and watermark images."""

import logging
import sys
from pprint import pformat
from time import perf_counter

from pyimg.args import parse_args
from pyimg.data_structures import Config
from pyimg.image import process_image
from pyimg.utils import get_summary_report

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

    LOG.debug("Runtime config:\n%s", pformat(cfg.as_dict(), indent=2))
    ctx = process_image(cfg)
    ctx.time_start = time_start
    LOG.debug(
        "Image Context:\n%s",
        pformat(ctx.as_dict(exclude_attrs=["image_buffer", "orig_exif"]), indent=2),
    )

    if not cfg.no_op:
        if not ctx.image_buffer:
            LOG.critical("Image buffer cannot be None")
            raise ValueError("Image buffer is None")
        LOG.info("Saving buffer to %s", cfg.output_file)
        with open(cfg.output_file, "wb") as f:
            f.write(ctx.image_buffer)
    else:
        print("***Displaying Results Only***")

    ctx.time_end = perf_counter()
    print(*get_summary_report(cfg, ctx), sep="\n")


if __name__ == "__main__":
    main()
