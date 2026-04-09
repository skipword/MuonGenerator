from pathlib import Path
from typing import Dict, Tuple

import torch

from nflows import distributions, flows, transforms

from ..core.settings import CONTEXT_COLS, DEVICE, MODEL_REGISTRY
from .helpers import load_json


def find_last_model(trial_dir: Path) -> Path:
    candidates = [
        trial_dir / "last_model",
        trial_dir / "last_model.pt",
        trial_dir / "last_model.pth",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = list(trial_dir.glob("*last_model*"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"No encontré last_model* en: {trial_dir}")


def find_stats_file(trial_dir: Path) -> Path:
    candidates = [
        trial_dir / "stats_minmax.json",
        trial_dir / "stats_angle.json",
        trial_dir / "stats.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No encontré stats*.json en {trial_dir}. "
        f"Crea stats_angle.json (o stats_minmax.json) con h,bx,bz "
        f"(y para energía: energy_log)."
    )


def build_flow_from_config(
    trial_dir: Path,
    context_features: int,
    device: torch.device,
) -> Tuple[flows.Flow, dict, dict]:
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config.json en {trial_dir}")

    cfg = load_json(config_path)

    hidden_features = int(cfg["hidden_features"])
    num_layers = int(cfg["num_layers"])
    num_bins = int(cfg["num_bins"])
    tail_bound = float(cfg["tail_bound"])

    min_bin_width = float(cfg.get("min_bin_width", 1e-3))
    min_bin_height = float(cfg.get("min_bin_height", 1e-4))
    min_derivative = float(cfg.get("min_derivative", 1e-3))

    layers = [
        transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            context_features=context_features,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        for _ in range(num_layers)
    ]

    transform = transforms.CompositeTransform(layers)
    base_dist = distributions.StandardNormal(shape=[1])
    flow = flows.Flow(transform, base_dist).to(device)

    model_path = find_last_model(trial_dir)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    flow.load_state_dict(state_dict)
    flow.eval()

    stats_path = find_stats_file(trial_dir)
    stats = load_json(stats_path)

    return flow, cfg, stats


class ModelBundle:
    def __init__(
        self,
        name: str,
        trial_dir: Path,
        flow: flows.Flow,
        cfg: dict,
        stats: dict,
        device: torch.device,
    ):
        self.name = name
        self.trial_dir = trial_dir
        self.flow = flow
        self.cfg = cfg
        self.stats = stats
        self.device = device


def load_all_models() -> Dict[str, ModelBundle]:
    bundles: Dict[str, ModelBundle] = {}

    for name, trial_dir in MODEL_REGISTRY.items():
        flow, cfg, stats = build_flow_from_config(
            trial_dir=trial_dir,
            context_features=len(CONTEXT_COLS),
            device=DEVICE,
        )
        bundles[name] = ModelBundle(name, trial_dir, flow, cfg, stats, DEVICE)

    return bundles