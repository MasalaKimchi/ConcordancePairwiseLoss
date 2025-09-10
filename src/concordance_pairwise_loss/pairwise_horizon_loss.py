import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcordancePairwiseHorizonLoss(nn.Module):
    """Pairwise concordance surrogate with horizon weighting.

    All time quantities are assumed to already be in **years**. When
    ``horizon_kind`` is not ``"none"`` the training split median follow-up
    time (in years) must be registered via :meth:`set_train_stats` before the
    loss is used for training or evaluation.
    """

    def __init__(
        self,
        horizon_kind: str = "exp",
        rel_factor: float = 0.5,
        temperature: float = 1.0,
        hetero_tau: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if horizon_kind not in {"none", "exp", "gauss", "tri"}:
            raise ValueError("horizon_kind must be one of 'none', 'exp', 'gauss', 'tri'")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.horizon_kind = horizon_kind
        self.rel_factor = rel_factor
        self.temperature = temperature
        self.hetero_tau = hetero_tau
        self.reduction = reduction

        # Registered buffer for training statistics
        self.register_buffer("median_followup_years", torch.tensor(0.0))

    def set_train_stats(self, median_followup_years: float) -> None:
        """Store the training split median follow-up time in years.

        Args:
            median_followup_years: Median follow-up in years.
        """
        self.median_followup_years.copy_(torch.tensor(float(median_followup_years)))

    def forward(
        self,
        scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        log_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.dim() == 2 and scores.size(1) == 1:
            scores = scores.squeeze(1)
        elif scores.dim() != 1:
            raise ValueError("scores must be 1D or 2D with shape (N,1)")

        n = scores.size(0)
        if times.size(0) != n or events.size(0) != n:
            raise ValueError("times and events must match scores in the first dimension")

        device = scores.device
        times = times.to(device)
        events = events.to(device)

        if events.dtype != torch.bool:
            events = events.bool()

        # Comparable pair mask: t_i < t_j AND event_i
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)  # (n, n)
        comparable = (time_diff < 0) & events.unsqueeze(1).expand(-1, n)
        not_diag = ~torch.eye(n, dtype=torch.bool, device=device)
        comparable = comparable & not_diag & (time_diff != 0)

        if not torch.any(comparable):
            # Maintain gradient flow when no pairs exist
            return scores.sum() * 0.0

        # Pairwise margins
        margins = scores.unsqueeze(1) - scores.unsqueeze(0)  # (n, n)

        if self.hetero_tau:
            if log_tau is None:
                raise ValueError("log_tau must be provided when hetero_tau=True")
            if log_tau.dim() == 2 and log_tau.size(1) == 1:
                log_tau = log_tau.squeeze(1)
            tau = torch.exp(log_tau).clamp_min(1e-6)
            tau_i = tau.unsqueeze(1)
            tau_j = tau.unsqueeze(0)
            pair_temp = torch.sqrt(tau_i.pow(2) + tau_j.pow(2)).clamp_min(1e-6)
            margins = margins / pair_temp
        else:
            margins = margins / self.temperature

        pair_loss = F.softplus(-margins)

        # Horizon weighting
        if self.horizon_kind != "none" and self.median_followup_years.item() == 0.0:
            raise RuntimeError("set_train_stats must be called before forward when horizon_kind != 'none'")

        time_gap = torch.clamp(times.unsqueeze(0) - times.unsqueeze(1), min=0)
        if self.horizon_kind == "exp":
            h = torch.clamp(self.median_followup_years * self.rel_factor, 0.25, 5.0)
            horizon_w = torch.exp(-time_gap / h)
        elif self.horizon_kind == "gauss":
            h = torch.clamp(self.median_followup_years * self.rel_factor, 0.25, 5.0)
            horizon_w = torch.exp(-(time_gap**2) / (2 * h**2))
        elif self.horizon_kind == "tri":
            h = torch.clamp(self.median_followup_years * self.rel_factor, 0.25, 5.0)
            horizon_w = torch.clamp(1 - time_gap / (3 * h), min=0.0)
        else:  # "none"
            horizon_w = torch.ones_like(time_gap, device=device)
            h = torch.tensor(float('nan'), device=device)

        weights = horizon_w * comparable.float()

        numerator = (pair_loss * weights).sum()
        denom = weights.sum().clamp_min(1e-12)

        if self.reduction == "mean":
            loss = numerator / denom
        elif self.reduction == "sum":
            loss = numerator
        else:  # "none"
            loss = (pair_loss * weights).sum(dim=1)

        return loss

    @property
    def derived_h(self) -> torch.Tensor:
        """Current derived horizon scale ``h`` in years."""
        if self.horizon_kind == "none":
            return torch.tensor(float("nan"), device=self.median_followup_years.device)
        return torch.clamp(self.median_followup_years * self.rel_factor, 0.25, 5.0)
