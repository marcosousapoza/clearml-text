"""Output and plotting helpers for inspection workflows."""

from pathlib import Path
from typing import Any


def configure_plot_style() -> None:
    """Apply a consistent Matplotlib style for inspection figures.

    The function keeps plot defaults in one place so notebook sections render
    with the same DPI, spine visibility, and title emphasis.
    """

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def dataset_output_dir(
    dataset_name: str,
    base_dir: str | Path = Path("out") / "inspection",
) -> Path:
    """Return the output directory for a dataset and create it if needed."""

    out_dir = Path(base_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(
    fig: Any,
    *,
    dataset_name: str,
    stem: str,
    base_dir: str | Path = Path("out") / "inspection",
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Persist a Matplotlib figure under the dataset-specific inspection folder.

    Each format in ``formats`` is written to ``<base_dir>/<dataset_name>/<stem>``.
    The function returns every generated path so callers can report or reuse
    them later in the notebook.
    """

    written: list[Path] = []
    out_dir = dataset_output_dir(dataset_name, base_dir=base_dir)
    for fmt in formats:
        target = out_dir / f"{stem}.{fmt}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    return written
