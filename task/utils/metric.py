import numpy as np
import sklearn.metrics as skm
from numpy.typing import NDArray


def roc_auc(target: NDArray, pred: NDArray) -> float:
    if pred.ndim <= 1 or pred.shape[1] <= 1:
        raise ValueError("multiclass ROC AUC requires class prediction scores.")

    target = target.astype(np.int64, copy=False)
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