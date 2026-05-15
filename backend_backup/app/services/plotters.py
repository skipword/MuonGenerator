import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.settings import (
    BINS_E,
    E_Y_MAX,
    E_Y_MIN,
    SIM_DURATION_SECONDS,
    THETA_BINS,
)
from .helpers import get_energy_stats_key


# -----------------------------
# Estilo global de gráficas
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "lines.linewidth": 2.0
})


def save_energy_spectrum_plot(
    out_png,
    E: np.ndarray,
    scale: float,
    title: str,
    subtitle: str,
    stats: dict,
):
    energy_key = get_energy_stats_key(stats)
    e_min = float(stats[energy_key]["min"])
    e_max = float(stats[energy_key]["max"])
    z_lo = E_Y_MIN * (e_max - e_min) + e_min
    z_hi = E_Y_MAX * (e_max - e_min) + e_min
    E_lo = float(np.exp(z_lo))
    E_hi = float(np.exp(z_hi))

    counts, edges = np.histogram(E, bins=BINS_E)
    spec = (scale * counts) / SIM_DURATION_SECONDS
    spec = spec.astype(float)
    spec[spec <= 0] = np.nan

    ymin = np.nanmin(spec)
    ymax = np.nanmax(spec)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = 1e-6, 1.0

    plt.figure(figsize=(8, 10), dpi=130)
    plt.step(edges[:-1], spec, where="post")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(E_lo, E_hi)
    plt.ylim(ymin / 2.0, ymax * 2.0)
    plt.xlabel("Energía (GeV)")
    plt.ylabel("part/(m²·s)")
    plt.title(title)
    plt.text(
        0.02,
        0.02,
        subtitle,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_angle_spectrum_plot(
    out_png,
    theta_deg: np.ndarray,
    scale: float,
    title: str,
    subtitle: str,
):
    counts, _ = np.histogram(theta_deg, bins=THETA_BINS)
    spec = (scale * counts) / SIM_DURATION_SECONDS
    centers = 0.5 * (THETA_BINS[:-1] + THETA_BINS[1:])

    plt.figure(figsize=(8, 10), dpi=140)
    plt.step(centers, spec, where="mid")
    plt.xlabel("Ángulo cenital θ (grados)")
    plt.ylabel("part/(m²·s)")
    plt.title(title)
    plt.text(
        0.02,
        0.02,
        subtitle,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()