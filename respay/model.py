"""Final record-attention architecture and its training configuration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 70
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    huber_beta: float = 0.5
    mse_weight: float = 0.2
    lambda_case: float = 1.0
    lambda_group_consistency: float = 0.0
    lr_loss_weight: float = 0.5
    use_residual_target: bool = True


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResPayOutput(NamedTuple):
    prediction: torch.Tensor
    case_prediction: torch.Tensor
    lr_logit: torch.Tensor
    attention_weights: torch.Tensor


class ResPay(nn.Module):
    """Pool records within each payment unit and predict its update and LR score."""

    def __init__(self, numeric_state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.case_encoder = nn.Sequential(
            nn.Linear(numeric_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.group_stats_encoder = nn.Sequential(
            nn.Linear(numeric_state_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.group_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.case_pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.lr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _mean_pool(
        self, values: torch.Tensor, group_idx: torch.Tensor, num_groups: int
    ) -> torch.Tensor:
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        sums = torch.zeros(num_groups, values.size(-1), device=values.device, dtype=values.dtype)
        sums.index_add_(0, group_idx, values)
        counts = torch.bincount(group_idx, minlength=num_groups).to(values.device)
        return sums / counts.clamp_min(1).to(values.dtype).unsqueeze(-1)

    def _attention_pool(
        self, case_h: torch.Tensor, group_idx: torch.Tensor, num_groups: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_net(case_h).squeeze(-1)
        # Normalize scores within each group for stable, group-local attention.
        maxima = torch.full(
            (num_groups,), torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype
        )
        maxima.scatter_reduce_(0, group_idx, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - maxima[group_idx])
        denominators = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
        denominators.index_add_(0, group_idx, exp_scores)
        weights = exp_scores / denominators[group_idx].clamp_min(1e-12)
        pooled = torch.zeros(num_groups, case_h.size(-1), device=case_h.device, dtype=case_h.dtype)
        pooled.index_add_(0, group_idx, case_h * weights.unsqueeze(-1))
        return pooled, weights

    def forward(
        self,
        s_numeric: torch.Tensor,
        group_idx: torch.Tensor,
        num_groups: int | None = None,
    ) -> ResPayOutput:
        if num_groups is None:
            num_groups = int(group_idx.max().item()) + 1
        case_h = self.case_encoder(s_numeric)
        pooled, attention = self._attention_pool(case_h, group_idx, num_groups)
        mean = self._mean_pool(s_numeric, group_idx, num_groups)
        variance = self._mean_pool((s_numeric - mean[group_idx]).pow(2), group_idx, num_groups)
        std = torch.sqrt(variance.clamp_min(1e-8))
        counts = torch.bincount(group_idx, minlength=num_groups).clamp_min(1).to(s_numeric.dtype)
        summary = torch.cat([mean, std, torch.log1p(counts).unsqueeze(-1)], dim=-1)
        stats_h = self.group_stats_encoder(summary)
        fused = self.group_fusion(torch.cat([pooled, stats_h], dim=-1))
        case_prediction = self.case_pred_head(
            torch.cat([case_h, fused[group_idx]], dim=-1)
        ).squeeze(-1)
        case_mean = self._mean_pool(case_prediction, group_idx, num_groups).squeeze(-1)
        prediction = case_mean + self.pred_head(fused).squeeze(-1)
        return ResPayOutput(prediction, case_prediction, self.lr_head(fused).squeeze(-1), attention)


def regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, mse_weight: float = 0.2
) -> torch.Tensor:
    """Unit-level absolute error with the selected quadratic penalty."""
    error = prediction - target
    return error.abs().mean() + mse_weight * error.pow(2).mean()
