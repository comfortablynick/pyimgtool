"""Helper functions."""
from typing import List

from pyimgtool.data_structures import Config, Context


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


def get_summary_report(cfg: Config, ctx: Context) -> List[str]:
    """Create formatted report based on details in `cfg` and `ctx`.

    Parameters
    ----------
    - `cfg` Config object
    - `ctx` Context object

    Returns
    -------
    - List of strings of formatted report "table"

    """
    size_reduction_bytes = ctx.orig_file_size - ctx.new_file_size
    report_title = " Processing Summary "
    report_end = " End "
    report_arrow = "->"
    report = []
    report.append(["File Name:", cfg.input_file, report_arrow, cfg.output_file])
    report.append(
        ["File Dimensions:", str(ctx.orig_size), report_arrow, str(ctx.new_size)]
    )
    report.append(
        [
            "File Size:",
            humanize_bytes(ctx.orig_file_size),
            report_arrow,
            humanize_bytes(ctx.new_file_size),
        ]
    )
    report.append(
        [
            "Size Reduction:",
            f"{humanize_bytes(size_reduction_bytes)} "
            f"({(size_reduction_bytes/ctx.orig_file_size) * 100:2.1f}%)",
        ]
    )
    report.append(
        ["Processing Time:", f"{(ctx.time_end - ctx.time_start)*1000:.1f} ms"]
    )
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
    out.append(f"{report_title:{'-'}^{col0w + col1w + col2w + col3w + 1}}")
    for line in report:
        out.append(
            f"{line[0]:<{col0w}} {line[1]:{col1w}} {line[2]:{col2w}} {line[3]:{col3w}}"
        )
    out.append(f"{report_end:{'-'}^{col0w + col1w + col2w + col3w + 1}}")
    return out
