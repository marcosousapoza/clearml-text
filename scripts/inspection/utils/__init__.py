"""Focused helpers for inspection notebooks and scripts."""

from .composition import (
    degree_dist,
    e2o_degree_summary,
    event_type_counts,
    events_per_object_distribution,
    object_type_counts,
    objects_per_event_distribution,
    type_counts,
)
from .io import configure_plot_style, dataset_output_dir, save_figure
from .influence import (
    compute_object_trace_lengths,
    event_share_by_min_object_trace_length,
    object_share_above_trace_length,
    object_type_occupation_above_trace_length,
)
from .schema import describe_tables, inspect_attribute_dtypes
from .stability import summarize_interval_stability
from .temporal import event_histogram, hist, object_histogram
from .timing import (
    event_object_matrix,
    event_object_recent_matrix,
    event_oldest_k_matrix,
    event_recent_k_matrix,
)
from .window_sampling import (
    sample_window_graph_sizes,
    summarize_sampled_window_graph_sizes,
)
from .missingness import attribute_non_null_summary

__all__ = [
    "attribute_non_null_summary",
    "configure_plot_style",
    "degree_dist",
    "event_share_by_min_object_trace_length",
    "compute_object_trace_lengths",
    "object_share_above_trace_length",
    "object_type_occupation_above_trace_length",
    "dataset_output_dir",
    "describe_tables",
    "e2o_degree_summary",
    "event_histogram",
    "hist",
    "event_object_matrix",
    "event_object_recent_matrix",
    "event_oldest_k_matrix",
    "event_recent_k_matrix",
    "event_type_counts",
    "events_per_object_distribution",
    "inspect_attribute_dtypes",
    "object_histogram",
    "object_type_counts",
    "objects_per_event_distribution",
    "save_figure",
    "sample_window_graph_sizes",
    "summarize_sampled_window_graph_sizes",
    "summarize_interval_stability",
    "type_counts",
]
