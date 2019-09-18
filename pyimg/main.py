"""Resize and watermark images."""
import logging
import sys
from pprint import pformat

from pyimg.args import parse_args
from pyimg.data_structures import Config
from pyimg.image import process_image


logging.basicConfig(level=logging.DEBUG)
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

    LOG.debug("Runtime config:\n%s", pformat(cfg.__dict__, indent=2))

    ctx = process_image(cfg)

    LOG.debug("Image Context:\n%s", pformat(ctx.as_dict_copy(), indent=2))

    if not cfg.no_op:
        if not ctx.image_buffer:
            LOG.critical("Image buffer cannot be None")
            raise ValueError("Image buffer is None")
        LOG.info("Saving buffer to %s", cfg.output_file)
        with open(cfg.output_file, "wb") as f:
            f.write(ctx.image_buffer)


if __name__ == "__main__":
    main()
