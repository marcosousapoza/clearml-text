import numpy as np
import sklearn.metrics as skm


def multiclass_roc_auc(true: np.ndarray, pred: np.ndarray) -> float:
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
