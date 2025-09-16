import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import torchsurv's IPCW implementation
from torchsurv.stats.ipcw import get_ipcw


class ConcordancePairwiseHorizonLoss(nn.Module):
    """Pairwise concordance surrogate with horizon weighting and IPCW support.

    All time quantities are assumed to already be in **years**. When
    ``horizon_kind`` is not ``"none"`` the training split median follow-up
    time (in years) must be registered via :meth:`set_train_stats` before the
    loss is used for training or evaluation.
    
    The horizon weighting functions are:
    - "exp": exponential decay exp(-Δt / h)
    - "gauss": Gaussian kernel exp(-Δt² / (2h²)) where h is the standard deviation
    - "tri": triangular kernel max(0, 1 - Δt / (3h))
    
    where Δt is the time gap between pairs and h = median_followup * rel_factor.
    
    IPCW weights are applied to the earlier index (i) for each comparable pair (i,j)
    following Uno's method for handling right-censoring.
    """

    def __init__(
        self,
        horizon_kind: str = "exp",
        rel_factor: float = 0.5,
        temperature: float = 1.0,
        hetero_tau: bool = False,
        reduction: str = "mean",
        use_ipcw: bool = True,
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
        self.use_ipcw = use_ipcw

        # Registered buffers for training statistics and adaptive clamping
        self.register_buffer("median_followup_years", torch.tensor(0.0))
        self.register_buffer("h_min_years", torch.tensor(0.0))
        self.register_buffer("h_max_years", torch.tensor(0.0))

    def set_train_stats(self, median_followup_years: float, h_min_years: Optional[float] = None, h_max_years: Optional[float] = None) -> None:
        """Store training statistics and optional adaptive clamping bounds in years.

        Args:
            median_followup_years: Median follow-up in years.
            h_min_years: Optional minimum clamp bound for h (years).
            h_max_years: Optional maximum clamp bound for h (years).
        """
        self.median_followup_years.copy_(torch.tensor(float(median_followup_years)))
        if h_min_years is not None:
            self.h_min_years.copy_(torch.tensor(float(h_min_years)))
        if h_max_years is not None:
            self.h_max_years.copy_(torch.tensor(float(h_max_years)))

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

        # Get IPCW weights if needed
        if self.use_ipcw:
            try:
                if events.any():  # Only compute IPCW if there are events
                    # torchsurv's get_ipcw has device issues, so we compute on CPU then move to GPU
                    events_cpu = events.cpu()
                    times_cpu = times.cpu()
                    ipcw = get_ipcw(events_cpu, times_cpu)
                    # Move result back to the same device as scores
                    ipcw = ipcw.to(device)
                else:
                    ipcw = torch.ones(n, device=device)
            except Exception:
                ipcw = torch.ones(n, device=device)
        else:
            ipcw = torch.ones(n, device=device)

        # Comparable pair mask: t_i < t_j AND event_i
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)  # (n, n)
        comparable = (time_diff < 0) & events.unsqueeze(1).expand(-1, n)
        not_diag = ~torch.eye(n, dtype=torch.bool, device=device)
        comparable = comparable & not_diag

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
        # Determine adaptive clamp bounds if provided, else default to [0.25, 5.0]
        has_adaptive = (self.h_min_years.item() > 0.0) and (self.h_max_years.item() > 0.0) and (self.h_max_years.item() >= self.h_min_years.item())
        if has_adaptive:
            h_lo = self.h_min_years
            h_hi = self.h_max_years
        else:
            h_lo = torch.tensor(0.25, device=device)
            h_hi = torch.tensor(5.0, device=device)

        if self.horizon_kind == "exp":
            h = torch.clamp(self.median_followup_years * self.rel_factor, h_lo, h_hi)
            horizon_w = torch.exp(-time_gap / h)
        elif self.horizon_kind == "gauss":
            h = torch.clamp(self.median_followup_years * self.rel_factor, h_lo, h_hi)
            horizon_w = torch.exp(-(time_gap**2) / (2 * h**2))
        elif self.horizon_kind == "tri":
            h = torch.clamp(self.median_followup_years * self.rel_factor, h_lo, h_hi)
            horizon_w = torch.clamp(1 - time_gap / (3 * h), min=0.0)
        else:  # "none"
            horizon_w = torch.ones_like(time_gap, device=device)
            h = torch.tensor(float('nan'), device=device)

        # Apply IPCW weights - only to earlier index (i) as per Uno's IPCW
        # ipcw[i] applies to pair (i,j) where i is the earlier sample with event
        ipcw_weights = ipcw.unsqueeze(1).expand(-1, n)  # Shape: (n, n)
        
        # Combine horizon weights with IPCW weights
        weights = horizon_w * ipcw_weights * comparable.float()

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
        device = self.median_followup_years.device
        has_adaptive = (self.h_min_years.item() > 0.0) and (self.h_max_years.item() > 0.0) and (self.h_max_years.item() >= self.h_min_years.item())
        if has_adaptive:
            h_lo = self.h_min_years
            h_hi = self.h_max_years
        else:
            h_lo = torch.tensor(0.25, device=device)
            h_hi = torch.tensor(5.0, device=device)
        return torch.clamp(self.median_followup_years * self.rel_factor, h_lo, h_hi)
