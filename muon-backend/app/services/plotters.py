from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..core.settings import AREA_M2, BINS_E, SIM_DURATION_SECONDS, THETA_BINS


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "grid.linewidth": 0.9,
})


def _compute_rate_per_bin(counts: np.ndarray, scale: float) -> np.ndarray:
    """
    Convierte cuentas por bin a tasa escalada en part/(m²·s).
    """
    return counts.astype(np.float64) * float(scale) / (AREA_M2 * SIM_DURATION_SECONDS)


def _style_axes(ax, *, use_minor_ticks: bool = True) -> None:
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    if use_minor_ticks:
        ax.minorticks_on()
        ax.tick_params(which="minor", length=3)
        ax.tick_params(which="major", length=6)


def _draw_info_box(ax, info_text: str | None) -> None:
    """
    Dibuja un cuadro de información en la esquina superior derecha.
    """
    if not info_text:
        return

    ax.text(
        0.972,
        0.972,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        multialignment="left",
        fontsize=11.2,
        linespacing=1.3,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.6",
            linewidth=1.0,
            alpha=0.95,
        ),
    )


def _positive_window(values: np.ndarray):
    """
    Devuelve el primer y último índice con valor positivo.
    """
    idx = np.flatnonzero(np.asarray(values) > 0)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def _step_xy_from_edges(edges: np.ndarray, values: np.ndarray):
    """
    Construye las coordenadas x,y para dibujar una curva tipo step a partir
    de bordes de bins y valores por bin.
    """
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(values, 2)
    return x, y


def _plot_closed_step(ax, edges: np.ndarray, values: np.ndarray, log_y: bool = False) -> None:
    """
    Dibuja una curva tipo step y agrega líneas verticales hacia abajo
    al inicio y al final del tramo positivo.
    """
    window = _positive_window(values)
    if window is None:
        return

    i0, i1 = window

    vals = values[i0:i1 + 1]
    sub_edges = edges[i0:i1 + 2]

    # En log no existe y=0, así que usamos un fondo positivo pequeño.
    positive_vals = values[values > 0]
    if log_y:
        bottom_y = positive_vals.min() / 10.0
    else:
        bottom_y = 0.0

    x, y = _step_xy_from_edges(sub_edges, vals)
    ax.plot(x, y)

    # Línea vertical al inicio
    ax.vlines(sub_edges[0], bottom_y, vals[0], linewidth=2.0)

    # Línea vertical al final
    ax.vlines(sub_edges[-1], bottom_y, vals[-1], linewidth=2.0)

    if log_y:
        current_top = np.nanmax(vals) * 1.15
        ax.set_ylim(bottom=bottom_y, top=current_top)


def save_energy_spectrum_plot(
    out_png: str | Path,
    E,
    scale=1.0,
    title="Espectro energético",
    subtitle: str | None = None,
    stats=None,
):
    """
    Guarda el espectro energético.
    """
    E = np.asarray(E, dtype=np.float64).reshape(-1)

    counts, edges = np.histogram(E, bins=BINS_E)
    spectrum = _compute_rate_per_bin(counts, scale)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)

    _plot_closed_step(ax, edges, spectrum, log_y=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])

    ax.set_xlabel("Energía (GeV)")
    ax.set_ylabel("part/(m²·s)")
    ax.set_title(title, pad=10)

    _style_axes(ax, use_minor_ticks=True)
    _draw_info_box(ax, subtitle)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_angle_spectrum_plot(
    out_png: str | Path,
    theta_deg,
    scale=1.0,
    title="Espectro angular",
    subtitle: str | None = None,
):
    """
    Guarda el espectro angular.
    """
    theta_deg = np.asarray(theta_deg, dtype=np.float64).reshape(-1)

    counts, edges = np.histogram(theta_deg, bins=THETA_BINS)
    spectrum = _compute_rate_per_bin(counts, scale)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)

    _plot_closed_step(ax, edges, spectrum, log_y=False)

    ax.set_xlim(edges[0], edges[-1])

    ax.set_xlabel(r"Ángulo cenital ($\theta^\circ$)")
    ax.set_ylabel("part/(m²·s)")
    ax.set_title(title, pad=10)

    _style_axes(ax, use_minor_ticks=True)
    _draw_info_box(ax, subtitle)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)