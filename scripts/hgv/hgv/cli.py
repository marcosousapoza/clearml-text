import argparse
import json
from pathlib import Path

from .model import parse
from .viz import render


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an OCEL JSON file as a hypergraph visualization."
    )
    parser.add_argument("input", help="Path to the OCEL JSON file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (.png, .pdf, .svg, .pgf). If omitted, only shows the window.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Do not open an interactive window.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=15.0,
        help="Figure width in inches (default: 15).",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=0.5,
        help="Height per object row in inches (default: 0.5).",
    )

    args = parser.parse_args()

    with Path(args.input).open() as f:
        data = json.load(f)
    render(
        parse(data),
        output=args.output,
        show=not args.no_show,
        width=args.width,
        row_height=args.row_height,
    )
