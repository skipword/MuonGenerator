from typing import Sequence, Tuple

import numpy as np
import torch

from nflows import flows


@torch.no_grad()
def sample_truncated_single_context_robust(
    flow: flows.Flow,
    c1: torch.Tensor,
    n: int,
    y_min: float,
    y_max: float,
    batch: int,
    oversample_factor: int,
    max_loops: int,
    shuffle_accept: bool = True,
) -> Tuple[np.ndarray, float]:
    samples, acc = sample_truncated_multi_context_robust(
        flow=flow,
        c1=c1,
        n=n,
        bounds=[(y_min, y_max)],
        batch=batch,
        oversample_factor=oversample_factor,
        max_loops=max_loops,
        shuffle_accept=shuffle_accept,
    )
    return samples[:, 0], acc


@torch.no_grad()
def sample_truncated_multi_context_robust(
    flow: flows.Flow,
    c1: torch.Tensor,
    n: int,
    bounds: Sequence[Tuple[float, float]],
    batch: int,
    oversample_factor: int,
    max_loops: int,
    shuffle_accept: bool = True,
) -> Tuple[np.ndarray, float]:
    flow.eval()

    if c1.dim() == 1:
        c1 = c1.view(1, -1)
    if c1.size(0) != 1:
        c1 = c1[:1]

    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    features = len(bounds)

    out_cpu = []
    need = int(n)
    drawn = 0
    kept = 0
    loops = 0

    while need > 0:
        loops += 1
        if loops > max_loops:
            raise RuntimeError(
                f"Demasiados loops (>{max_loops}). drawn={drawn}, kept={kept}, need={need}. "
                f"Truncación muy estricta o contexto fuera de rango."
            )

        m = max(batch, min(batch * oversample_factor, need * oversample_factor))
        m = int(m)

        y = flow.sample(m, context=c1)

        if y.dim() == 3:
            y = y.squeeze(0)
        y = y.reshape(-1, features)

        drawn += y.size(0)

        mask = torch.ones(y.size(0), dtype=torch.bool, device=y.device)
        for idx, (lo, hi) in enumerate(bounds):
            mask &= (y[:, idx] >= lo) & (y[:, idx] <= hi)

        y = y[mask]
        if y.numel() == 0:
            continue

        if shuffle_accept and y.size(0) > 1:
            perm = torch.randperm(y.size(0), device=y.device)
            y = y[perm]

        take = min(need, y.size(0))
        out_cpu.append(y[:take].detach().cpu())
        kept += take
        need -= take

    acc = kept / drawn if drawn > 0 else float("nan")
    return torch.cat(out_cpu, dim=0).numpy(), float(acc)