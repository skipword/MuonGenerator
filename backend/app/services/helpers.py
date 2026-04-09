import json
from pathlib import Path
from typing import Tuple

import numpy as np

from ..core.settings import CONTEXT_COLS


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_angle_eps_mu(stats: dict, default: float = 1e-6) -> float:
    try:
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
    e_min = float(stats["energy_log"]["min"])
    e_max = float(stats["energy_log"]["max"])
    z = y * (e_max - e_min) + e_min
    return np.exp(z)


def mu_to_theta_deg(mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu.astype(np.float64), 0.0, 1.0)
    return np.rad2deg(np.arccos(mu)).astype(np.float32)


def pdf_theta_deg_from_pdf_mu(
    grid_theta_deg: np.ndarray,
    pdf_mu_vals: np.ndarray,
) -> np.ndarray:
    th_rad = np.deg2rad(grid_theta_deg.astype(np.float64))
    return (pdf_mu_vals * np.sin(th_rad) * (np.pi / 180.0)).astype(np.float64)