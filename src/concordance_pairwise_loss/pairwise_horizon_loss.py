import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcordancePairwiseHorizonLoss(nn.Module):
    """Pairwise concordance surrogate with relative horizon weighting.

    All time quantities passed to :meth:`forward` **must already be in years**.
    When ``horizon_kind`` is anything other than ``"none"`` the caller must
    first invoke :meth:`set_train_stats` with the training-split median
    follow-up time in years. For the Gaussian kernel, ``h`` is used as the
    width parameter of ``exp(-(delta_t / h)**2)``.
    """

    def __init__(
        self,
        horizon_kind: str = "exp",
        rel_factor: float = 0.5,
        temperature: float = 1.0,
        hetero_tau: bool = False,
        reduction: str = "mean",
        tau_reg_weight: float = 0.0,
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
        self.tau_reg_weight = tau_reg_weight

        # Registered buffer for training statistics
        self.register_buffer("median_followup_years", torch.tensor(0.0))

    def set_train_stats(self, median_followup_years: float) -> None:
        """Store the training split median follow-up time in years.

        Args:
            median_followup_years: Median follow-up in years.
        """
        self.median_followup_years.copy_(
            torch.tensor(
                float(median_followup_years),
                device=self.median_followup_years.device,
            )
        )

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

        if self.horizon_kind != "none" and self.median_followup_years.item() == 0.0:
            raise ValueError(
                "set_train_stats(median_followup_years) must be called before training or "
                "evaluation when horizon_kind != 'none', and times must be in years."
            )

        # Comparable pair mask: include (i,j) iff times[i] < times[j] and event[i]
        time_i = times.unsqueeze(1)
        time_j = times.unsqueeze(0)
        comparable = (time_i < time_j) & events.unsqueeze(1)

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
            tau = torch.exp(log_tau).clamp_min(1e-3)
            tau_i = tau.unsqueeze(1)
            tau_j = tau.unsqueeze(0)
            pair_temp = torch.sqrt(tau_i.pow(2) + tau_j.pow(2)).clamp_min(1e-3)
            margins = margins / pair_temp
        else:
            margins = margins / self.temperature

        pair_loss = F.softplus(-margins)

        # Horizon weighting
        delta_t = torch.clamp(time_j - time_i, min=0)
        if self.horizon_kind == "exp":
            h = self.derived_h
            horizon_w = torch.exp(-delta_t / h)
        elif self.horizon_kind == "gauss":
            h = self.derived_h
            horizon_w = torch.exp(-(delta_t / h) ** 2)
        elif self.horizon_kind == "tri":
            h = self.derived_h
            horizon_w = torch.clamp(1 - delta_t / (3 * h), min=0.0)
        else:  # "none"
            horizon_w = torch.ones_like(delta_t, device=device)
            h = torch.tensor(float("nan"), device=device)

        weights = horizon_w * comparable.float()

        numerator = (pair_loss * weights).sum()
        denom = weights.sum().clamp_min(1e-8)

        if self.reduction == "mean":
            loss = numerator / denom
        elif self.reduction == "sum":
            loss = numerator
        else:  # "none"
            loss = (pair_loss * weights).sum(dim=1)

        if self.hetero_tau and self.tau_reg_weight > 0.0 and log_tau is not None:
            loss = loss + self.tau_reg_weight * (log_tau**2).mean()

        return loss

    @property
    def derived_h(self) -> torch.Tensor:
        """Current derived horizon scale ``h`` in years."""
        if self.horizon_kind == "none":
            return torch.tensor(float("nan"), device=self.median_followup_years.device)
        return torch.clamp(self.median_followup_years * self.rel_factor, 0.25, 5.0)
