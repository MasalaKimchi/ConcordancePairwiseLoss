import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from concordance_pairwise_loss.pairwise_horizon_loss import ConcordancePairwiseHorizonLoss

def test_pair_properties():
    scores = torch.tensor([0.0, 0.5, -0.5])
    times = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1, 0, 1])

    loss_fn = ConcordancePairwiseHorizonLoss(horizon_kind="exp")
    loss_fn.set_train_stats(median_followup_years=2.0)

    n = times.size(0)
    time_i = times.unsqueeze(1)
    time_j = times.unsqueeze(0)
    mask = (time_i < time_j) & events.unsqueeze(1)
    cmp_count = mask.sum().item()

    expected = 0
    for i in range(n):
        if events[i]:
            expected += (times > times[i]).sum().item()
    assert cmp_count == expected

    base_loss = loss_fn(scores, times, events)

    # Recompute weights and pairwise losses to verify scaling invariance
    margins = scores.unsqueeze(1) - scores.unsqueeze(0)
    margins = margins / loss_fn.temperature
    pair_loss = torch.nn.functional.softplus(-margins)
    delta_t = torch.clamp(time_j - time_i, min=0)
    h = torch.clamp(loss_fn.median_followup_years * loss_fn.rel_factor, 0.25, 5.0)
    horizon_w = torch.exp(-delta_t / h)
    weights = horizon_w * mask.float()
    mean1 = (pair_loss * weights).sum() / weights.sum()
    mean2 = (pair_loss * (2 * weights)).sum() / (2 * weights).sum()
    assert torch.allclose(mean1, mean2, atol=1e-6)

    # Increasing positive margin decreases the loss
    scores2 = scores.clone()
    scores2[0] += 1.0
    loss2 = loss_fn(scores2, times, events)
    assert loss2 < base_loss
