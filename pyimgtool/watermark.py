"""Watermark image with text or another image."""

import logging
import os
from datetime import datetime
from pathlib import PurePath

from PIL import Image, ImageDraw, ImageFont, ImageStat

from pyimgtool.data_structures import Config, Context, ImageSize, Position
from pyimgtool.resize import resize_height

LOG = logging.getLogger(__name__)


def get_region_stats(im: Image, region: list) -> ImageStat:
    """Get ImageStat object for region of PIL image.

    Parameters
    ----------
    - `image` The image to get the luminance of
    - `region` The region to get the luminance of,
               in the form [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]

    """
    image_l = im.convert("L")
    mask = Image.new("L", image_l.size, 0)
    drawing_layer = ImageDraw.Draw(mask)
    drawing_layer.rectangle(region, fill=255)
    return ImageStat.Stat(image_l, mask=mask)


def find_best_location(im: Image, size: ImageSize, padding: float) -> Position:
    """Find the best location for the watermark.

    The best location is the one with least luminance variance.

    Parameters
    ----------
    - `im` PIL Image
    - `size` Size of watermark image
    - `padding` Proportion of padding to add around watermark

    """
    bl_padding = tuple(
        map(
            lambda x: int(x),
            [padding * im.size[0], im.size[1] - size.height - padding * im.size[1]],
        )
    )
    br_padding = tuple(
        map(
            lambda x: int(x),
            [
                im.size[0] - size.width - padding * im.size[0],
                im.size[1] - size.height - padding * im.size[1],
            ],
        )
    )
    tl_padding = tuple(
        map(lambda x: int(x), [padding * im.size[0], padding * im.size[0]])
    )
    tr_padding = tuple(
        map(
            lambda x: int(x),
            [im.size[0] - size.width - padding * im.size[0], padding * im.size[1]],
        )
    )
    bc_padding = tuple(
        map(
            lambda x: int(x),
            [
                im.size[0] / 2 - size.width / 2,
                im.size[1] - size.height - padding * im.size[1],
            ],
        )
    )
    paddings = [bl_padding, br_padding, tl_padding, tr_padding, bc_padding]
    vars = list(
        map(
            lambda padding: get_region_stats(
                im, [padding, (padding[0] + size.width, padding[1] + size.height)]
            ).stddev[0],
            paddings,
        )
    )
    minimum = min(vars)
    index = vars.index(minimum)
    locations = [
        Position.BOTTOM_LEFT,
        Position.BOTTOM_RIGHT,
        Position.TOP_LEFT,
        Position.TOP_RIGHT,
        Position.BOTTOM_CENTER,
    ]
    return locations[index]


def add_copyright_date(im: Image, cfg: Config, ctx: Context) -> None:
    """Extract date taken from photo to add to copyright text.

    Parameters
    ----------
    - `im` PIL Image
    - `cfg` Config object
    - `ctx` Context object

    """
    photo_dt = datetime.now()
    if ctx.orig_exif is not None:
        try:
            photo_dt = datetime.strptime(
                ctx.orig_exif["Exif"]["DateTimeOriginal"].decode("utf-8"),
                "%Y:%m:%d %H:%M:%S",
            )
        except KeyError:
            pass
    copyright_year = photo_dt.strftime("%Y")
    cfg.text = f"© {copyright_year} {cfg.text_copyright}"
    LOG.info("Using copyright text: %s", cfg.text)
    LOG.info("Photo date from exif: %s", photo_dt)


def with_image(im: Image, cfg: Config, ctx: Context) -> Image:
    """Watermark with image according to Config.

    Parameters
    ----------
    - `im` PIL Image
    - `cfg` Config object
    - `ctx` Context object

    """
    if cfg.watermark_image is None:
        LOG.error("Missing watermark_image in cfg")
        return im
    watermark_image = Image.open(os.path.expanduser(cfg.watermark_image)).convert(
        "RGBA"
    )
    LOG.info("Watermark: %s", watermark_image.size)
    watermark_ratio = watermark_image.height / im.height
    LOG.info("Watermark size ratio: %.4f", watermark_ratio)
    if watermark_ratio > cfg.watermark_scale:
        LOG.debug(
            "Resizing watermark from %.4f to %.4f scale",
            watermark_ratio,
            cfg.watermark_scale,
        )
        watermark_image = resize_height(
            watermark_image,
            (int(im.width * cfg.watermark_scale), int(im.height * cfg.watermark_scale)),
        )
        LOG.debug("New watermark dims: %s", watermark_image.size)
    offset_x = cfg.watermark_padding
    offset_y = cfg.watermark_padding
    ctx.watermark_size = ImageSize(watermark_image.width, watermark_image.height)
    mask = watermark_image.split()[3].point(lambda i: i * cfg.watermark_opacity)
    pos = (
        (im.width - watermark_image.width - offset_x),
        (im.height - watermark_image.height - offset_y),
    )
    loc = find_best_location(im, ctx.watermark_size, 0.05)
    LOG.debug("Best detected watermark loc: %s", loc)
    im.paste(watermark_image, pos, mask)
    return im


def with_text(im: Image, cfg: Config, ctx: Context) -> Image:
    """Watermark with text if program option is supplied.

    If text is equal to 'copyright', the exif data will be read
    (if available) to attempt to determine copyright date based on
    date photo was taken.

    Parameters
    ----------
    - `im` PIL Image
    - `cfg` Config object
    - `ctx` Context object

    """
    if cfg.text is None and cfg.text_copyright is None:
        LOG.error("Missing text or copyright text in cfg")
        return im
    if cfg.text_copyright is not None:
        # Add date photo taken to copyright text
        add_copyright_date(im, cfg, ctx)
    layer = Image.new("RGBA", (im.width, im.height), (255, 255, 255, 0))

    font_size = 1  # starting size
    offset_x = cfg.text_padding
    offset_y = cfg.text_padding

    try:
        cwd = PurePath(os.path.dirname(__file__))
        font_path = str(PurePath.joinpath(cwd, "fonts", "SourceSansPro-Regular.ttf"))
        font = ImageFont.truetype(font=font_path, size=font_size)
    except OSError:
        LOG.error("Could not find font '%s', aborting text watermark", font_path)
        return im

    LOG.debug("Found font '%s'", font_path)
    while font.getsize(cfg.text)[0] < cfg.text_scale * im.width:
        # iterate until text size is >= text_scale
        font_size += 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    if font.getsize(cfg.text)[0] > cfg.text_scale * im.width:
        font_size -= 1
        font = ImageFont.truetype(font=font_path, size=font_size)

    text_width, text_height = font.getsize(cfg.text)
    LOG.debug(
        "Final text dims: %d x %d px; Font size: %d", text_width, text_height, font_size
    )
    # TODO: calculate watermark dims accurately
    stats = get_region_stats(
        im, [offset_x, offset_y, text_width, im.height - text_height]
    )
    LOG.debug("Region luminance: %f", stats.mean[0])
    LOG.debug("Region luminance stddev: %f", stats.stddev[0])
    d = ImageDraw.Draw(layer)
    opacity = int(round((cfg.text_opacity * 255)))
    LOG.info("Text opacity: %d/255", opacity)

    text_fill = 255, 255, 255, opacity
    if stats.mean[0] / 256 >= 0.5:
        text_fill = 0, 0, 0, opacity

    d.text((offset_x, offset_y), cfg.text, font=font, fill=text_fill)
    out = Image.alpha_composite(im.convert("RGBA"), layer)
    return out.convert("RGB")
