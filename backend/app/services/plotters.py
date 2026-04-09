import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nflows import flows

from ..core.settings import (
    BINS_E,
    DEVICE,
    E_Y_MAX,
    E_Y_MIN,
    SIM_DURATION_SECONDS,
    THETA_BINS,
    THETA_BIN_WIDTH,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
)
from .helpers import pdf_theta_deg_from_pdf_mu


plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "lines.linewidth": 2.0,
})


def save_energy_spectrum_plot(
    out_png,
    E: np.ndarray,
    scale: float,
    title: str,
    subtitle: str,
    stats: dict,
):
    e_min = float(stats["energy_log"]["min"])
    e_max = float(stats["energy_log"]["max"])
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
    plt.step(edges[:-1], spec, where="post", linewidth=2.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(E_lo, E_hi)
    plt.ylim(ymin / 2.0, ymax * 2.0)
    plt.xlabel("Energy (GeV)", fontsize=13)
    plt.ylabel("part/(m²·s)", fontsize=13)
    plt.title(title, fontsize=14)
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


def calcular_pdf_mu(flow, context_vec, grid_mu, device, chunk=2000):
    flow.eval()
    pdf_list = []

    with torch.no_grad():
        for i in range(0, len(grid_mu), chunk):
            g = grid_mu[i:i + chunk].astype(np.float32)
            g_t = torch.from_numpy(g).unsqueeze(1).to(device)
            ctx = context_vec.repeat(g_t.size(0), 1)
            logp = flow.log_prob(g_t, context=ctx).detach().cpu().numpy()
            pdf_list.append(np.exp(logp))

    pdf_mu = np.concatenate(pdf_list, axis=0).reshape(-1)
    return np.clip(pdf_mu, 1e-30, None)


def save_angle_spectrum_plot(
    out_png,
    theta_deg: np.ndarray,
    scale: float,
    rate_total: float,
    flow: flows.Flow,
    context_vec,
    eps_mu: float,
    title: str,
    subtitle: str,
):
    counts, _ = np.histogram(theta_deg, bins=THETA_BINS)
    spec = (scale * counts) / SIM_DURATION_SECONDS
    centers = 0.5 * (THETA_BINS[:-1] + THETA_BINS[1:])

    grid_th = np.linspace(THETA_MIN_DEG, THETA_MAX_DEG, 1200, dtype=np.float64)
    mu_from_th = np.cos(np.deg2rad(grid_th))
    mu_from_th = np.clip(mu_from_th, eps_mu, 1.0 - eps_mu)

    pdf_mu = calcular_pdf_mu(flow, context_vec, mu_from_th, device=DEVICE, chunk=2000)
    pdf_th_deg = pdf_theta_deg_from_pdf_mu(grid_th, pdf_mu)
    _ = rate_total * pdf_th_deg * THETA_BIN_WIDTH

    plt.figure(figsize=(8, 10), dpi=140)
    plt.step(centers, spec, where="mid", linewidth=2.2, label="Model (hist)")
    plt.xlabel("θ (degrees)")
    plt.ylabel("part/(m²·s)")
    plt.title(title, fontsize=14)
    plt.text(
        0.02,
        0.02,
        subtitle,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


import torch