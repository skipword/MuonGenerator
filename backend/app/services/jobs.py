from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..core.settings import (
    ANGLE_EPS_DEFAULT,
    AREA_M2,
    CAP_EVENTS,
    E_Y_MAX,
    E_Y_MIN,
    MAX_LOOPS,
    MUON_MASS_GEV,
    OVERSAMPLE_FACTOR,
    PZ_SIGN,
    SAMPLE_BATCH,
    SIM_DURATION_SECONDS,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
)
from ..schemas import SimFullRequest, SimRequest
from .helpers import (
    ctx_to_mm,
    get_angle_eps_mu,
    get_mu_encoding,
    mm_to_energy,
    mu_to_theta_deg,
)
from .model_loader import ModelBundle
from .physics import (
    flux_per_m2_per_s,
    momentum_components_from_angles,
    momentum_from_total_energy_GeV,
    sample_phi_deg_uniform,
    target_counts,
)
from .plotters import save_angle_spectrum_plot, save_energy_spectrum_plot
from .sampling import sample_truncated_multi_context_robust
from .writers import (
    make_zip,
    write_angle_csv,
    write_angle_shw,
    write_energy_csv,
    write_energy_shw,
    write_full_csv,
    write_full_shw,
)

SimulationRequest = SimRequest | SimFullRequest


def _build_context_tensor(
    bundle: ModelBundle,
    altura: float,
    bx: float,
    bz: float,
) -> tuple[np.ndarray, dict, torch.Tensor]:
    context_mm, clip_info = ctx_to_mm(altura, bx, bz, bundle.stats, clip=True)
    context_tensor = torch.tensor(
        context_mm,
        dtype=torch.float32,
        device=bundle.device,
    ).view(1, -1)
    return context_mm, clip_info, context_tensor


def _compute_run_counts(altura: float) -> tuple[float, float, int, int]:
    flux = flux_per_m2_per_s(altura)
    n_target = target_counts(altura, SIM_DURATION_SECONDS, AREA_M2)
    n_target_int = int(max(0, round(n_target)))
    n_draw = int(min(n_target_int, CAP_EVENTS))

    if n_draw <= 0:
        raise RuntimeError("N_draw <= 0. Revisa altura y fórmula de flujo.")

    return flux, n_target, n_target_int, n_draw


def _write_meta_json(out_meta: Path, meta: dict) -> None:
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _build_energy_meta(
    *,
    bundle: ModelBundle,
    req: SimRequest,
    flux: float,
    n_target_int: int,
    n_draw: int,
    scale: float,
    acceptance: float,
    clip_info: dict,
    mu_encoding: str,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "energy",
        "modelo": req.modelo,
        "bundle": bundle.name,
        "trial_dir": str(bundle.trial_dir),
        "features": int(bundle.features),
        "mu_encoding": mu_encoding,
        "bx_uT": float(req.bx),
        "bz_uT": float(req.bz),
        "altura_m": float(req.altura),
        "flux_part_m2_s": float(flux),
        "duration_s": float(SIM_DURATION_SECONDS),
        "area_m2": float(AREA_M2),
        "N_target": int(n_target_int),
        "N_draw": int(n_draw),
        "scale": float(scale),
        "acceptance": float(acceptance),
        "E_Y_MIN": float(bundle.cfg.get("y_min", E_Y_MIN)),
        "E_Y_MAX": float(bundle.cfg.get("y_max", E_Y_MAX)),
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }


def _build_angle_meta(
    *,
    bundle: ModelBundle,
    req: SimRequest,
    flux: float,
    rate_total: float,
    n_target_int: int,
    n_draw: int,
    scale: float,
    acceptance: float,
    mu_encoding: str,
    clip_info: dict,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "angle",
        "modelo": req.modelo,
        "bundle": bundle.name,
        "trial_dir": str(bundle.trial_dir),
        "features": int(bundle.features),
        "mu_encoding": mu_encoding,
        "bx_uT": float(req.bx),
        "bz_uT": float(req.bz),
        "altura_m": float(req.altura),
        "flux_part_m2_s": float(flux),
        "duration_s": float(SIM_DURATION_SECONDS),
        "area_m2": float(AREA_M2),
        "rate_total_part_m2_s": float(rate_total),
        "N_target": int(n_target_int),
        "N_draw": int(n_draw),
        "scale": float(scale),
        "acceptance": float(acceptance),
        "mu_min": float(bundle.cfg.get("mu_min", get_angle_eps_mu(bundle.stats, bundle.cfg, ANGLE_EPS_DEFAULT))),
        "mu_max": float(bundle.cfg.get("mu_max", 1.0 - get_angle_eps_mu(bundle.stats, bundle.cfg, ANGLE_EPS_DEFAULT))),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }


def _build_full_meta(
    *,
    bundle: ModelBundle,
    req: SimFullRequest,
    flux: float,
    rate_total: float,
    n_target_int: int,
    n_draw: int,
    scale: float,
    acceptance: float,
    mu_encoding: str,
    n_below_mass: int,
    clip_info: dict,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "full",
        "bundle": bundle.name,
        "trial_dir": str(bundle.trial_dir),
        "features": int(bundle.features),
        "sampling_mode": "joint_2d",
        "mu_encoding": mu_encoding,
        "bx_uT": float(req.bx),
        "bz_uT": float(req.bz),
        "altura_m": float(req.altura),
        "flux_part_m2_s": float(flux),
        "duration_s": float(SIM_DURATION_SECONDS),
        "area_m2": float(AREA_M2),
        "rate_total_part_m2_s": float(rate_total),
        "N_target": int(n_target_int),
        "N_draw": int(n_draw),
        "scale": float(scale),
        "acceptance": float(acceptance),
        "E_Y_MIN": float(bundle.cfg.get("y_min", E_Y_MIN)),
        "E_Y_MAX": float(bundle.cfg.get("y_max", E_Y_MAX)),
        "mu_min": float(bundle.cfg.get("mu_min", get_angle_eps_mu(bundle.stats, bundle.cfg, ANGLE_EPS_DEFAULT))),
        "mu_max": float(bundle.cfg.get("mu_max", 1.0 - get_angle_eps_mu(bundle.stats, bundle.cfg, ANGLE_EPS_DEFAULT))),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "phi_distribution": "uniform_deg_[0,360)",
        "muon_mass_GeV": float(MUON_MASS_GEV),
        "pz_sign_convention": float(PZ_SIGN),
        "n_energy_below_muon_mass": int(n_below_mass),
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }


def sample_joint_events(
    bundle: ModelBundle,
    altura: float,
    bx: float,
    bz: float,
    n_draw: int,
    n_target_int: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict, torch.Tensor, str]:
    if bundle.features != 2:
        raise RuntimeError(
            f"El modelo cargado debe tener 2 features (energía y mu). Encontrado: {bundle.features}."
        )

    mu_encoding = get_mu_encoding(bundle.stats, bundle.cfg)
    mu_eps = get_angle_eps_mu(bundle.stats, bundle.cfg, ANGLE_EPS_DEFAULT)

    e_y_min = float(bundle.cfg.get("y_min", E_Y_MIN))
    e_y_max = float(bundle.cfg.get("y_max", E_Y_MAX))
    mu_min = float(bundle.cfg.get("mu_min", mu_eps))
    mu_max = float(bundle.cfg.get("mu_max", 1.0 - mu_eps))

    _, clip_info, context_tensor = _build_context_tensor(bundle, altura, bx, bz)

    joint_samples, acceptance = sample_truncated_multi_context_robust(
        flow=bundle.flow,
        c1=context_tensor,
        n=n_draw,
        bounds=[(e_y_min, e_y_max), (mu_min, mu_max)],
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    energy_mm = np.asarray(joint_samples[:, 0], dtype=np.float64).reshape(-1)
    mu_samples = np.asarray(joint_samples[:, 1], dtype=np.float64).reshape(-1)
    energy_samples = np.asarray(mm_to_energy(energy_mm, bundle.stats), dtype=np.float64).reshape(-1)
    theta_deg = mu_to_theta_deg(mu_samples, encoding=mu_encoding).reshape(-1)
    scale = (n_target_int / n_draw) if n_draw > 0 else 1.0

    return (
        energy_samples,
        theta_deg,
        mu_samples,
        scale,
        acceptance,
        clip_info,
        context_tensor,
        mu_encoding,
    )


def simulate_energy_job(
    bundle: ModelBundle,
    req: SimRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    flux, _, n_target_int, n_draw = _compute_run_counts(req.altura)

    (
        energy_samples,
        _,
        _,
        scale,
        acceptance,
        clip_info,
        _,
        mu_encoding,
    ) = sample_joint_events(
        bundle,
        req.altura,
        req.bx,
        req.bz,
        n_draw,
        n_target_int,
    )

    out_png = run_dir / "image.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"

    elapsed_s = time.perf_counter() - request_start_perf
    subtitle = (
        f"numero_particulas={n_draw} | "
        f"tiempo_flujo={SIM_DURATION_SECONDS:g}s | tiempo_simulado={elapsed_s:.2f}s"
    )

    save_energy_spectrum_plot(
        out_png,
        energy_samples,
        scale,
        "Espectro de energía (CNF)",
        subtitle,
        stats=bundle.stats,
    )

    meta = _build_energy_meta(
        bundle=bundle,
        req=req,
        flux=flux,
        n_target_int=n_target_int,
        n_draw=n_draw,
        scale=scale,
        acceptance=acceptance,
        clip_info=clip_info,
        mu_encoding=mu_encoding,
        elapsed_s=elapsed_s,
    )

    _write_meta_json(out_meta, meta)

    write_energy_csv(out_csv, energy_samples, meta)
    write_energy_shw(out_shw, energy_samples, meta)
    make_zip(out_zip, [out_shw])

    return meta


def simulate_angle_job(
    bundle: ModelBundle,
    req: SimRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    flux, _, n_target_int, n_draw = _compute_run_counts(req.altura)

    (
        _,
        theta_deg,
        _,
        scale,
        acceptance,
        clip_info,
        _,
        mu_encoding,
    ) = sample_joint_events(
        bundle,
        req.altura,
        req.bx,
        req.bz,
        n_draw,
        n_target_int,
    )

    mask_plot = (theta_deg >= THETA_MIN_DEG) & (theta_deg <= THETA_MAX_DEG)
    theta_plot = theta_deg[mask_plot]

    out_png = run_dir / "image.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"

    elapsed_s = time.perf_counter() - request_start_perf
    rate_total = n_target_int / SIM_DURATION_SECONDS

    subtitle = (
        f"numero_particulas={n_draw} | "
        f"tiempo_flujo={SIM_DURATION_SECONDS:g}s | tiempo_simulado={elapsed_s:.2f}s"
    )

    save_angle_spectrum_plot(
        out_png=out_png,
        theta_deg=theta_plot,
        scale=scale,
        title="Espectro Angular (CNF)",
        subtitle=subtitle,
    )

    meta = _build_angle_meta(
        bundle=bundle,
        req=req,
        flux=flux,
        rate_total=rate_total,
        n_target_int=n_target_int,
        n_draw=n_draw,
        scale=scale,
        acceptance=acceptance,
        mu_encoding=mu_encoding,
        clip_info=clip_info,
        elapsed_s=elapsed_s,
    )

    _write_meta_json(out_meta, meta)

    write_angle_csv(out_csv, theta_plot, meta)
    write_angle_shw(out_shw, theta_plot, meta)
    make_zip(out_zip, [out_shw])

    return meta


def simulate_full_job(
    bundle: ModelBundle,
    req: SimFullRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    flux, _, n_target_int, n_draw = _compute_run_counts(req.altura)

    (
        energy_samples,
        theta_deg,
        _,
        scale,
        acceptance,
        clip_info,
        _,
        mu_encoding,
    ) = sample_joint_events(
        bundle,
        req.altura,
        req.bx,
        req.bz,
        n_draw,
        n_target_int,
    )

    phi_deg = sample_phi_deg_uniform(n_draw)
    p_gev_c = momentum_from_total_energy_GeV(energy_samples)
    px_gev_c, py_gev_c, pz_gev_c = momentum_components_from_angles(
        p_gev_c,
        theta_deg,
        phi_deg,
    )

    mask_plot = (theta_deg >= THETA_MIN_DEG) & (theta_deg <= THETA_MAX_DEG)
    theta_plot = theta_deg[mask_plot]

    out_energy_png = run_dir / "energy.png"
    out_angle_png = run_dir / "angle.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"

    elapsed_s = time.perf_counter() - request_start_perf
    rate_total = n_target_int / SIM_DURATION_SECONDS

    subtitle_energy = (
        f"numero_particulas={n_draw} | "
        f"tiempo_flujo={SIM_DURATION_SECONDS:g}s | tiempo_simulado={elapsed_s:.2f}s"
    )
    subtitle_angle = (
        f"numero_particulas={n_draw} | "
        f"tiempo_flujo={SIM_DURATION_SECONDS:g}s | tiempo_simulado={elapsed_s:.2f}s"
    )

    save_energy_spectrum_plot(
        out_png=out_energy_png,
        E=energy_samples,
        scale=scale,
        title="Espectro de Energía (CNF)",
        subtitle=subtitle_energy,
        stats=bundle.stats,
    )

    save_angle_spectrum_plot(
        out_png=out_angle_png,
        theta_deg=theta_plot,
        scale=scale,
        title="Espectro Angular (CNF)",
        subtitle=subtitle_angle,
    )

    n_below_mass = int(np.sum(energy_samples < MUON_MASS_GEV))

    meta = _build_full_meta(
        bundle=bundle,
        req=req,
        flux=flux,
        rate_total=rate_total,
        n_target_int=n_target_int,
        n_draw=n_draw,
        scale=scale,
        acceptance=acceptance,
        mu_encoding=mu_encoding,
        n_below_mass=n_below_mass,
        clip_info=clip_info,
        elapsed_s=elapsed_s,
    )

    _write_meta_json(out_meta, meta)

    write_full_csv(
        out_csv=out_csv,
        energy_GeV=energy_samples,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        p_GeV_c=p_gev_c,
        px_GeV_c=px_gev_c,
        py_GeV_c=py_gev_c,
        pz_GeV_c=pz_gev_c,
        meta=meta,
    )

    write_full_shw(
        out_shw=out_shw,
        energy_GeV=energy_samples,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        p_GeV_c=p_gev_c,
        px_GeV_c=px_gev_c,
        py_GeV_c=py_gev_c,
        pz_GeV_c=pz_gev_c,
        meta=meta,
    )

    make_zip(out_zip, [out_shw])

    return meta