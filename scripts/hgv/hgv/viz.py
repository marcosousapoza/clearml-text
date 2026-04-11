import os
from datetime import datetime, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from .model import OcelLog


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

# Tableau-10-inspired palette, slightly desaturated for print/research use.
# Falls back to cycling if more than 10 types are needed.
_PALETTE = [
    (0.306, 0.475, 0.655),  # steel blue
    (0.882, 0.341, 0.349),  # muted coral
    (0.349, 0.631, 0.310),  # sage green
    (0.949, 0.557, 0.169),  # warm amber
    (0.690, 0.478, 0.631),  # soft lavender
    (0.463, 0.718, 0.698),  # teal
    (0.929, 0.788, 0.282),  # straw yellow
    (0.612, 0.459, 0.373),  # warm brown
    (1.000, 0.616, 0.655),  # blush pink
    (0.729, 0.690, 0.675),  # warm grey
]


def _make_palette(n: int) -> list[tuple[float, float, float]]:
    """Return n colors by cycling through the curated research palette."""
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _lighten(color: tuple, factor: float = 0.25) -> tuple:
    """Blend an RGB color toward white. factor=0 → white, factor=1 → original."""
    r, g, b = color
    return (r * factor + (1 - factor), g * factor + (1 - factor), b * factor + (1 - factor))


# ---------------------------------------------------------------------------
# Color dict helpers (public)
# ---------------------------------------------------------------------------

def make_color_dicts(
    log: OcelLog,
) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """Return ``(obj_color, evt_color)`` dicts mapping type names → RGB tuples.

    Passing these explicitly to :func:`render` ensures consistent colours
    across multiple subplots that may show different subsets of the same log.
    """
    obj_colors = _make_palette(len(log.object_types))
    evt_colors = _make_palette(len(log.event_types))
    return (
        dict(zip(log.object_types, obj_colors)),
        dict(zip(log.event_types,  evt_colors)),
    )


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _format_tick(seconds: float, epoch: datetime, total_span: float) -> str:
    """Human-readable tick label scaled to the span."""
    target = epoch + timedelta(seconds=seconds)
    if total_span <= 120:
        return target.strftime("%H:%M:%S")
    elif total_span <= 7200:
        return target.strftime("%H:%M")
    elif total_span <= 86400 * 2:
        return target.strftime("%m-%d %H:%M")
    else:
        return target.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Legend helper (public)
# ---------------------------------------------------------------------------

def draw_legend(
    log: OcelLog,
    ax_leg: Axes,
    obj_color: dict[str, tuple],
    evt_color: dict[str, tuple],
) -> None:
    """Draw a combined object-type + event-type legend into *ax_leg*.

    Intended for use when multiple panels share a single legend.  Pass the
    ``obj_color`` / ``evt_color`` dicts returned by :func:`make_color_dicts`
    so the colours match the rendered panels exactly.
    """
    sec_header = dict(facecolor="none", edgecolor="none")

    obj_handles = [
        mpatches.Patch(facecolor=_lighten(obj_color[t], 0.30),
                       edgecolor=obj_color[t], linewidth=1.2, label=t)
        for t in log.object_types if t in obj_color
    ]
    evt_handles = [
        mpatches.Patch(color=evt_color[t], label=t)
        for t in log.event_types if t in evt_color
    ]

    all_handles: list = []
    if obj_handles:
        all_handles += [mpatches.Patch(label="Object types", **sec_header)] + obj_handles
    if evt_handles:
        if all_handles:
            all_handles += [mpatches.Patch(label=" ", **sec_header)]
        all_handles += [mpatches.Patch(label="Event types", **sec_header)] + evt_handles

    if not all_handles:
        return

    leg = ax_leg.legend(
        handles=all_handles,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.98),
        bbox_transform=ax_leg.transAxes,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=1.4,
        handletextpad=0.5,
        borderpad=0.7,
    )
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        if handle.get_facecolor()[3] == 0 and handle.get_edgecolor()[3] == 0:
            text.set_fontweight("bold")
            text.set_fontsize(7.5)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

HALF_BAND = 0.38   # default half-height of each row band in data units
BAND_GAP  = 0.04   # default gap between adjacent bands (each side)

def render(
    log: OcelLog,
    *,
    width: float = 15.0,
    row_height: float = 0.7,
    band_size: float = HALF_BAND,
    band_gap: float = BAND_GAP,
    output: str | None = None,
    show: bool = True,
    ax: Axes | None = None,
    obj_color: dict[str, tuple] | None = None,
    evt_color: dict[str, tuple] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
) -> Figure:
    """Render an OcelLog as a hypergraph visualization.

    Parameters
    ----------
    log:        Parsed OCEL log.
    width:      Figure width in inches (ignored when *ax* is provided).
    row_height: Height per object row in inches (ignored when *ax* is provided).
    band_size:  Half-height of each object row band in data units (rows are
                spaced 1 unit apart, so 0.5 would fill the full row).
                Default ``0.38`` leaves a small visual gap between rows.
    band_gap:   Additional inset on each side of the band where the hyperedge
                line protrudes beyond the band edge.  Default ``0.04``.
    output:     File path to save (.png, .pdf, .svg, .pgf). None = no save.
    show:       Whether to call plt.show().
    ax:         If given, draw into this existing axes instead of creating a
                new figure.  Legend, save, and show are all skipped.
    obj_color:  Mapping of object-type name → RGB colour.  When supplied the
                same palette is used across multiple panels.
    evt_color:  Mapping of event-type name → RGB colour.  Same as above.
    time_range: Optional ``(t_min, t_max)`` pair of datetimes that fixes the
                x-axis span.  When provided, all plots using the same range
                are directly comparable.  Defaults to the log's own extent.
    """
    # ------------------------------------------------------------------
    # 1. Sort objects: group by type (preserving objectTypes order), then id
    # ------------------------------------------------------------------
    type_rank = {t: i for i, t in enumerate(log.object_types)}
    sorted_objects = sorted(log.objects, key=lambda o: (type_rank.get(o.type, 999), o.id))
    obj_row: dict[str, int] = {o.id: i for i, o in enumerate(sorted_objects)}
    n_rows = len(sorted_objects)

    # ------------------------------------------------------------------
    # 2. Time range & x mapping
    # ------------------------------------------------------------------
    def _strip_tz(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

    if time_range is not None:
        t_min = _strip_tz(time_range[0])
        t_max = _strip_tz(time_range[1])
    else:
        times = [e.time for e in log.events]
        t_min = _strip_tz(min(times)) if times else datetime(2000, 1, 1)
        t_max = _strip_tz(max(times)) if times else datetime(2000, 1, 1)

    span = max((t_max - t_min).total_seconds(), 1.0)

    x_width = 12.0  # data-coordinate width of the drawing area
    x_pad = x_width * 0.05  # 5% buffer on each side

    # ------------------------------------------------------------------
    # 3. Colors — use provided dicts or derive from this log
    # ------------------------------------------------------------------
    if obj_color is None or evt_color is None:
        _obj, _evt = make_color_dicts(log)
        if obj_color is None:
            obj_color = _obj
        if evt_color is None:
            evt_color = _evt

    # ------------------------------------------------------------------
    # 4. Figure & axes
    # ------------------------------------------------------------------
    if ax is None:
        margin_top    = 0.8
        margin_bottom = 1.6
        fig_height = max(3.5, n_rows * row_height + margin_top + margin_bottom)

        LEGEND_W = 1.8   # fixed inches for the legend panel
        main_w   = max(1.0, width - LEGEND_W)

        fig = plt.figure(figsize=(width, fig_height))
        gs  = GridSpec(1, 2, figure=fig,
                       width_ratios=[main_w, LEGEND_W],
                       wspace=0.03)
        ax     = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        ax_leg.axis("off")
        draw_legend(log, ax_leg, obj_color, evt_color)
        _owns_figure = True
    else:
        fig = ax.figure
        _owns_figure = False

    label_margin = 1.2   # data units reserved for object-id labels on the left
    ax.set_xlim(-label_margin, x_width + 0.6)
    ax.set_ylim(-(n_rows - 1) - 1.0, 1.0)
    ax.axis("off")
    ax.set_aspect("auto")

    # ------------------------------------------------------------------
    # 5. Object row bands + labels
    # ------------------------------------------------------------------
    band_inner = band_size - band_gap   # inner half-height (slightly smaller for gap)

    for obj in sorted_objects:
        row = obj_row[obj.id]
        y_center = -row
        base_color  = obj_color.get(obj.type, (0.5, 0.5, 0.5))
        fill_color  = _lighten(base_color, factor=0.30)
        edge_color  = _lighten(base_color, factor=0.55)

        # Filled band (rounded rectangle)
        band = FancyBboxPatch(
            (0, y_center - band_inner),
            x_width,
            2 * band_inner,
            boxstyle="round,pad=0.02",
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(band)

        # Object id label to the left, colored with the base color
        ax.text(
            -0.15, y_center, obj.id,
            ha="right", va="center",
            fontsize=8, color=base_color,
            fontweight="semibold",
        )

    # ------------------------------------------------------------------
    # 6. Event hyperedge vertical lines + participation circles
    # ------------------------------------------------------------------
    for event in log.events:
        x = (
            (_strip_tz(event.time) - t_min).total_seconds()
            / span
            * (x_width - 2 * x_pad)
            + x_pad
        )
        color = evt_color.get(event.type, (0.25, 0.25, 0.25))

        participating_rows = [obj_row[oid] for oid in event.object_ids if oid in obj_row]

        if participating_rows:
            y_top = -min(participating_rows) + band_inner
            y_bot = -max(participating_rows) - band_inner
            ax.vlines(
                x, y_bot, y_top,
                colors=[color], linewidths=2.2, alpha=0.90, zorder=3,
            )
            for row in participating_rows:
                ax.plot(
                    x, -row,
                    "o",
                    markersize=9,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=5,
                )
        else:
            # No participating objects: tick below the last row
            ax.plot(
                x, -(n_rows - 1) - 0.6,
                "|", color=color, markersize=10, markeredgewidth=2.0, zorder=4,
            )

    # ------------------------------------------------------------------
    # 7. Time axis
    # ------------------------------------------------------------------
    axis_y = -(n_rows - 1) - 0.72

    ax.hlines(axis_y, 0, x_width, colors=["#444444"], linewidths=1.2, zorder=2)
    ax.annotate(
        "",
        xy=(x_width + 0.4, axis_y),
        xytext=(x_width, axis_y),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
    )

    n_ticks = min(8, max(2, len(log.events)))
    tick_xs = (
        [x_width * i / (n_ticks - 1) for i in range(n_ticks)]
        if n_ticks > 1 else [0.0, x_width]
    )
    for tx in tick_xs:
        t_sec = tx / x_width * span
        label = _format_tick(t_sec, t_min, span)
        ax.plot(tx, axis_y, "|", color="#444444", markersize=6, markeredgewidth=1.2, zorder=3)
        ax.text(
            tx, axis_y - 0.18, label,
            ha="center", va="top",
            fontsize=7, color="#444444", rotation=30,
        )

    # ------------------------------------------------------------------
    # 8. Save / show  (only when we own the figure)
    # ------------------------------------------------------------------
    if _owns_figure:
        fig.tight_layout(pad=0.4)

        if output:
            ext = os.path.splitext(output)[1].lower()
            if ext == ".pgf":
                fig.savefig(output, backend="pgf", bbox_inches="tight")
            else:
                fig.savefig(output, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

    return fig
