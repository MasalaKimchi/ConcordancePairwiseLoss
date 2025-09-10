import torch
import torch.nn as nn
from typing import Callable, Optional

from .pairwise_horizon_loss import ConcordancePairwiseHorizonLoss


class UncertaintyWeightedCombination(nn.Module):
    """Combine losses using learned uncertainty weighting."""

    def __init__(
        self,
        rank_loss: ConcordancePairwiseHorizonLoss,
        disc_time_nll_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.rank_loss = rank_loss
        self.disc_time_nll_fn = disc_time_nll_fn
        # log variances for rank loss and NLL loss
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, scores, times, events, log_tau=None, **kwargs):
        L_rank = self.rank_loss(scores, times, events, log_tau=log_tau)
        L_nll = (
            self.disc_time_nll_fn(
                scores=scores, times=times, events=events, **kwargs
            )
            if self.disc_time_nll_fn is not None
            else None
        )
        if L_nll is not None:
            w_rank = torch.exp(-2.0 * self.log_vars[0])
            w_nll = torch.exp(-2.0 * self.log_vars[1])
            return w_rank * L_rank + w_nll * L_nll + self.log_vars.sum()
        return L_rank

    @property
    def rank_weight(self) -> torch.Tensor:
        return torch.exp(-2.0 * self.log_vars[0]).detach()

    @property
    def nll_weight(self) -> torch.Tensor:
        return torch.exp(-2.0 * self.log_vars[1]).detach()
