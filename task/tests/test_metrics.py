import numpy as np

from task.metrics import roc_auc


def test_roc_auc_supports_binary_probability_vector() -> None:
    target = np.array([0, 1, 0, 1], dtype=np.int64)
    pred = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)

    score = roc_auc(target, pred)

    assert score == 1.0


def test_roc_auc_supports_binary_two_column_scores() -> None:
    target = np.array([0, 1, 0, 1], dtype=np.int64)
    pred = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )

    score = roc_auc(target, pred)

    assert score == 1.0


def test_roc_auc_supports_multiclass_scores() -> None:
    target = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    pred = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.7, 0.1],
            [0.75, 0.15, 0.1],
            [0.05, 0.15, 0.8],
        ],
        dtype=np.float64,
    )

    score = roc_auc(target, pred)

    assert score == 1.0
