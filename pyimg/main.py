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
    """Start point."""
    cfg = Config.from_args(parse_args(sys.argv[1:]))
    log_level = 0
    try:
        log_level = (0, 20, 10)[cfg.verbosity]
    except IndexError:
        log_level = 10
    #  LOG.setLevel(log_level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for l in loggers:
        l.setLevel(log_level)
    LOG.debug("Runtime config:\n%s", pformat(cfg.__dict__, indent=2))

    outbuf = process_image(cfg)

    if not cfg.no_op:
        LOG.info("Saving buffer to %s", cfg.output_file)
        with open(cfg.output_file, "wb") as f:
            f.write(outbuf)


if __name__ == "__main__":
    main()
