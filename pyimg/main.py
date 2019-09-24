"""Resize and watermark images."""

import logging
import sys
from pprint import pformat

from pyimg.args import parse_args
from pyimg.data_structures import Config
from pyimg.image import process_image

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger(__name__)


def main():
    """Process image based on config file and command-line arguments."""
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
    LOG.debug("Image Context:\n%s", pformat(ctx.as_dict(), indent=2))

    # Create report

    report_title = " Summary "
    report_end = " End "
    report = []
    report.append(["File Name:", cfg.input_file, "->", cfg.output_file])
    report.append(["File Dimensions:", str(ctx.orig_size), "->", str(ctx.new_size)])
    for c in report:
        c[2] = "" if c[3] == c[1] else c[2]
        c[3] = "  " if c[3] == c[1] else c[3]

    padding = 2
    col0w = max([len(c[0]) for c in report]) + padding
    col1w = max([len(c[1]) for c in report]) + padding
    col2w = max([len(c[2]) for c in report]) + padding
    col3w = max([len(c[3]) for c in report]) + padding
    print(f"{report_title:{'-'}^{col0w + col1w + col2w + col3w + 1}}")
    for line in report:
        print(
            f"{line[0]:<{col0w}} {line[1]:{col1w}} {line[2]:{col2w}} {line[3]:{col3w}}"
        )
    print(f"{report_end:{'-'}^{col0w + col1w + col2w + col3w + 1}}")

    if not cfg.no_op:
        if not ctx.image_buffer:
            LOG.critical("Image buffer cannot be None")
            raise ValueError("Image buffer is None")
        LOG.info("Saving buffer to %s", cfg.output_file)
        with open(cfg.output_file, "wb") as f:
            f.write(ctx.image_buffer)
    else:
        print("***Displaying Results Only***")


if __name__ == "__main__":
    main()
