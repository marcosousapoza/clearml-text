import numpy as np
import sklearn.metrics as skm
from numpy.typing import NDArray


# roc_auc: weighted per-class OvR, used as a metric callable in task `metrics` lists.
# multiclass_roc_auc: macro OvR with softmax normalization, for external callers.


def roc_auc(target: NDArray, pred: NDArray) -> float:
    """ROC AUC for binary or multiclass predictions.

    For binary tasks, ``pred`` may be a 1-D score vector or a two-column score
    matrix; in the latter case the positive-class column is used. For
    multiclass tasks, this computes a weighted one-vs-rest ROC AUC and skips
    classes with no positives or no negatives.
    """
    target = target.astype(np.int64, copy=False)

    if pred.ndim == 1:
        try:
            return float(skm.roc_auc_score(target, pred))
        except ValueError:
            return float("nan")

    if pred.shape[1] == 1:
        try:
            return float(skm.roc_auc_score(target, pred[:, 0]))
        except ValueError:
            return float("nan")

    if pred.shape[1] == 2:
        try:
            return float(skm.roc_auc_score(target, pred[:, 1]))
        except ValueError:
            return float("nan")

    scores: list[float] = []
    weights: list[int] = []
    for label in range(pred.shape[1]):
        binary_target = target == label
        positives = int(binary_target.sum())
        negatives = len(binary_target) - positives
        if positives == 0 or negatives == 0:
            continue
        scores.append(float(skm.roc_auc_score(binary_target, pred[:, label])))
        weights.append(positives)

    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def multiclass_roc_auc(true: np.ndarray, pred: np.ndarray) -> float:
    """Macro one-vs-rest ROC AUC with softmax normalization.

    Converts raw logits to probabilities before scoring. Returns nan when
    fewer than two classes are present in the ground truth.
    """
    if pred.ndim <= 1 or pred.shape[1] <= 1:
        raise ValueError("multiclass_roc_auc requires class prediction scores.")
    shifted = pred - pred.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob = prob / prob.sum(axis=1, keepdims=True)
    try:
        labels = np.unique(true.astype(np.int64, copy=False))
        if labels.size < 2:
            return float("nan")
        if labels.size == 2:
            positive_label = int(labels[-1])
            binary_true = (true == positive_label).astype(np.int64, copy=False)
            return float(skm.roc_auc_score(binary_true, prob[:, positive_label]))
        labels = np.arange(pred.shape[1], dtype=int)
        return float(
            skm.roc_auc_score(
                true,
                prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def multiclass_log_loss(true: np.ndarray, pred: np.ndarray) -> float:
    """Cross-entropy loss for multiclass predictions, given raw logits."""
    if pred.ndim <= 1 or pred.shape[1] <= 1:
        raise ValueError("multiclass_log_loss requires class prediction scores.")
    shifted = pred - pred.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob = prob / prob.sum(axis=1, keepdims=True)
    try:
        if pred.shape[1] == 2:
            return float(skm.log_loss(true, prob[:, 1]))
        labels = np.arange(pred.shape[1], dtype=int)
        return float(skm.log_loss(true, prob, labels=labels))
    except ValueError:
        return float("nan")
