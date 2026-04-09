import math
from typing import Tuple

import numpy as np

from ..core.settings import (
    AZIMUTH_MAX_DEG,
    AZIMUTH_MIN_DEG,
    FLUX_A,
    FLUX_B,
    MUON_MASS_GEV,
    PZ_SIGN,
)


def flux_per_m2_per_s(h: float) -> float:
    return FLUX_A * math.exp(FLUX_B * float(h))


def target_counts(h: float, duration_s: float, area_m2: float) -> float:
    return flux_per_m2_per_s(h) * float(duration_s) * float(area_m2)


def sample_phi_deg_uniform(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(AZIMUTH_MIN_DEG, AZIMUTH_MAX_DEG, size=int(n)).astype(np.float64)


def momentum_from_total_energy_GeV(energy_GeV: np.ndarray) -> np.ndarray:
    E = np.asarray(energy_GeV, dtype=np.float64).reshape(-1)
    p2 = np.maximum(E * E - MUON_MASS_GEV * MUON_MASS_GEV, 0.0)
    return np.sqrt(p2)


def momentum_components_from_angles(
    p_GeV_c: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p_GeV_c, dtype=np.float64).reshape(-1)
    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=np.float64).reshape(-1))
    phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64).reshape(-1))

    px = p * np.sin(theta_rad) * np.cos(phi_rad)
    py = p * np.sin(theta_rad) * np.sin(phi_rad)
    pz = PZ_SIGN * p * np.cos(theta_rad)

    return px, py, pz