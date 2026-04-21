import json
from pathlib import Path
from typing import Tuple

import numpy as np

from ..core.settings import CONTEXT_COLS


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_energy_stats_key(stats: dict) -> str:
    if "logE" in stats:
        return "logE"
    if "energy_log" in stats:
        return "energy_log"
    raise KeyError("No se encontró 'logE' ni 'energy_log' en stats.")


def get_mu_encoding(stats: dict, cfg: dict | None = None) -> str:
    texts = []

    if isinstance(cfg, dict):
        for key in ("target", "mu_target", "note"):
            value = cfg.get(key)
            if isinstance(value, str):
                texts.append(value.lower())

    for key in ("angle_mu", "angle_target"):
        value = stats.get(key)
        if isinstance(value, dict):
            for subkey in ("type", "target", "transform"):
                subvalue = value.get(subkey)
                if isinstance(subvalue, str):
                    texts.append(subvalue.lower())

    joined = " | ".join(texts)
    if any(token in joined for token in ("1-cos", "1 - cos", "1−cos", "1− cos", "1 -cos")):
        return "one_minus_cos"
    if "cos(theta" in joined or "cos(theta_deg" in joined:
        return "cos"

    if "logE" in stats and "mu" in stats:
        return "one_minus_cos"

    return "cos"


def get_angle_eps_mu(stats: dict, cfg: dict | None = None, default: float = 1e-6) -> float:
    try:
        if isinstance(cfg, dict):
            for key in ("mu_min", "eps_mu", "eps_clip"):
                if key in cfg:
                    return float(cfg[key])

        if isinstance(stats.get("angle_mu"), dict) and "mu_clip_eps" in stats["angle_mu"]:
            return float(stats["angle_mu"]["mu_clip_eps"])
        if isinstance(stats.get("angle_target"), dict) and "eps_clip" in stats["angle_target"]:
            return float(stats["angle_target"]["eps_clip"])
    except Exception:
        pass
    return float(default)


def _minmax(v: float, vmin: float, vmax: float) -> float:
    return (v - vmin) / (vmax - vmin)


def ctx_to_mm(
    altura: float,
    bx: float,
    bz: float,
    stats: dict,
    clip: bool = True,
) -> Tuple[np.ndarray, dict]:
    raw = {"h": float(altura), "bx": float(bx), "bz": float(bz)}
    vec = []
    clip_info = {}

    for k in CONTEXT_COLS:
        vmin = float(stats[k]["min"])
        vmax = float(stats[k]["max"])
        mm = _minmax(raw[k], vmin, vmax)
        mm0 = mm

        if clip:
            mm = float(np.clip(mm, 0.0, 1.0))

        vec.append(mm)

        if mm != mm0:
            clip_info[k] = mm0

    return np.array(vec, dtype=np.float32), clip_info


def mm_to_energy(y: np.ndarray, stats: dict) -> np.ndarray:
    energy_key = get_energy_stats_key(stats)
    e_min = float(stats[energy_key]["min"])
    e_max = float(stats[energy_key]["max"])
    z = y * (e_max - e_min) + e_min
    return np.exp(z)


def theta_deg_to_mu(theta_deg: np.ndarray, encoding: str = "cos") -> np.ndarray:
    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    cos_theta = np.cos(theta_rad)

    if encoding == "one_minus_cos":
        mu = 1.0 - cos_theta
    else:
        mu = cos_theta

    return np.clip(mu, 0.0, 1.0).astype(np.float32)


def mu_to_theta_deg(mu: np.ndarray, encoding: str = "cos") -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)

    if encoding == "one_minus_cos":
        arg = np.clip(1.0 - mu, -1.0, 1.0)
    else:
        arg = np.clip(mu, -1.0, 1.0)

    return np.rad2deg(np.arccos(arg)).astype(np.float32)


def pdf_theta_deg_from_pdf_mu(
    grid_theta_deg: np.ndarray,
    pdf_mu_vals: np.ndarray,
) -> np.ndarray:
    th_rad = np.deg2rad(grid_theta_deg.astype(np.float64))
    return (pdf_mu_vals * np.sin(th_rad) * (np.pi / 180.0)).astype(np.float64)