from __future__ import annotations

from typing import Literal

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _SKQuantileTransformer


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


class QuantileTargetTransform(TargetTransform):
    """Wraps sklearn's QuantileTransformer to map targets to a normal distribution."""

    def __init__(self, n_quantiles: int = 1000, output_distribution: Literal["uniform", "normal"] = "normal") -> None:
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def fit(self, target: torch.Tensor, y=None):
        self._qt = _SKQuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            subsample=int(1e9),
        )
        self._qt.fit(target.float().numpy().reshape(-1, 1))
        return self

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        transformed = self._qt.transform(target.float().numpy().reshape(-1, 1))
        return torch.from_numpy(transformed.reshape(-1)).float()

    def inverse_transform(self, target: torch.Tensor) -> torch.Tensor:
        inversed = self._qt.inverse_transform(target.float().numpy().reshape(-1, 1))
        return torch.from_numpy(inversed.reshape(-1)).float()
