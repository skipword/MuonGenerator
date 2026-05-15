from pathlib import Path
from typing import Dict, Tuple

import torch

from nflows import distributions, flows, transforms

from ..core.settings import CONTEXT_COLS, DEVICE, MODEL_REGISTRY
from .helpers import load_json


def find_model_file(trial_dir: Path) -> Path:
    candidates = [
        trial_dir / "model.pt",
        trial_dir / "model.pth",
        trial_dir / "last_model",
        trial_dir / "last_model.pt",
        trial_dir / "last_model.pth",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = list(trial_dir.glob("*model*")) + list(trial_dir.glob("*last_model*"))
    hits = [p for p in hits if p.is_file()]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"No encontré model.pt/last_model* en: {trial_dir}")


def find_stats_file(trial_dir: Path) -> Path:
    candidates = [
        trial_dir / "stats.json",
        trial_dir / "stats_minmax.json",
        trial_dir / "stats_angle.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"No encontré stats*.json en {trial_dir}.")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def infer_feature_count(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith("autoregressive_net.initial_layer.weight"):
            return int(value.shape[1])
    raise RuntimeError("No pude inferir el número de features desde el checkpoint.")


def count_permutations(state_dict: dict) -> int:
    return sum(1 for key in state_dict if key.endswith("._permutation"))


def build_flow_from_config(
    trial_dir: Path,
    context_features: int,
    device: torch.device,
) -> Tuple[flows.Flow, dict, dict, int]:
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

    model_path = find_model_file(trial_dir)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    features = infer_feature_count(state_dict)
    permutation_count = count_permutations(state_dict)

    layers = []
    for idx in range(num_layers):
        layers.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                context_features=context_features,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        )
        if permutation_count > idx and features > 1:
            layers.append(transforms.ReversePermutation(features=features))

    transform = transforms.CompositeTransform(layers)
    base_dist = distributions.StandardNormal(shape=[features])
    flow = flows.Flow(transform, base_dist).to(device)
    flow.load_state_dict(state_dict)
    flow.eval()

    stats_path = find_stats_file(trial_dir)
    stats = load_json(stats_path)

    return flow, cfg, stats, features


class ModelBundle:
    def __init__(
        self,
        name: str,
        trial_dir: Path,
        flow: flows.Flow,
        cfg: dict,
        stats: dict,
        device: torch.device,
        features: int,
    ):
        self.name = name
        self.trial_dir = trial_dir
        self.flow = flow
        self.cfg = cfg
        self.stats = stats
        self.device = device
        self.features = features


def load_all_models() -> Dict[str, ModelBundle]:
    bundles: Dict[str, ModelBundle] = {}

    for name, trial_dir in MODEL_REGISTRY.items():
        flow, cfg, stats, features = build_flow_from_config(
            trial_dir=trial_dir,
            context_features=len(CONTEXT_COLS),
            device=DEVICE,
        )
        bundles[name] = ModelBundle(name, trial_dir, flow, cfg, stats, DEVICE, features)

    return bundles