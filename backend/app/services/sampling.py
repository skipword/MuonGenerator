from typing import Tuple

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
    flow.eval()

    if c1.dim() == 1:
        c1 = c1.view(1, -1)
    if c1.size(0) != 1:
        c1 = c1[:1]

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
        y = y.reshape(-1)

        drawn += y.numel()

        mask = (y >= y_min) & (y <= y_max)
        y = y[mask]
        if y.numel() == 0:
            continue

        if shuffle_accept and y.numel() > 1:
            perm = torch.randperm(y.numel(), device=y.device)
            y = y[perm]

        take = min(need, y.numel())
        out_cpu.append(y[:take].detach().cpu())
        kept += take
        need -= take

    acc = kept / drawn if drawn > 0 else float("nan")
    return torch.cat(out_cpu, dim=0).numpy(), float(acc)