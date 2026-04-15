from __future__ import annotations

import torch
from sklearn.base import BaseEstimator, TransformerMixin


class TargetTransform(BaseEstimator, TransformerMixin):
    """Identity transform — a no-op base class for target normalization.

    Wire in a subclass by overriding make_target_transform() on a task:
      - ZScoreTargetTransform: standardizes to zero mean and unit variance
      - Log1pZScoreTargetTransform: log1p first, then Z-score; good for
        skewed regression targets like time durations
    """

    def fit(self, target: torch.Tensor, y=None):
        return self

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def inverse_transform(self, target: torch.Tensor) -> torch.Tensor:
        return target


class ZScoreTargetTransform(TargetTransform):
    def fit(self, target: torch.Tensor, y=None):
        target = target.float()
        std = target.std(unbiased=False)
        self.mean_ = float(target.mean())
        self.std_ = float(std) if torch.isfinite(std) and std > 0 else 1.0
        return self

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return (target.float() - self.mean_) / self.std_

    def inverse_transform(self, target: torch.Tensor) -> torch.Tensor:
        return target.float() * self.std_ + self.mean_


class Log1pZScoreTargetTransform(ZScoreTargetTransform):
    def fit(self, target: torch.Tensor, y=None):
        return super().fit(torch.log1p(target.float()), y)

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return super().transform(torch.log1p(target.float()))

    def inverse_transform(self, target: torch.Tensor) -> torch.Tensor:
        return torch.expm1(super().inverse_transform(target))
