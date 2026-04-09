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
from .helpers import ctx_to_mm, get_angle_eps_mu, mm_to_energy, mu_to_theta_deg
from .model_loader import ModelBundle
from .physics import (
    flux_per_m2_per_s,
    momentum_components_from_angles,
    momentum_from_total_energy_GeV,
    sample_phi_deg_uniform,
    target_counts,
)
from .plotters import save_angle_spectrum_plot, save_energy_spectrum_plot
from .sampling import sample_truncated_single_context_robust
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
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "energy",
        "modelo": req.modelo,
        "bundle": bundle.name,
        "trial_dir": str(bundle.trial_dir),
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
        "E_Y_MIN": float(E_Y_MIN),
        "E_Y_MAX": float(E_Y_MAX),
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
    eps_mu: float,
    clip_info: dict,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "angle",
        "modelo": req.modelo,
        "bundle": bundle.name,
        "trial_dir": str(bundle.trial_dir),
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
        "eps_mu": float(eps_mu),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }


def _build_full_meta(
    *,
    bundle_energy: ModelBundle,
    bundle_angle: ModelBundle,
    req: SimFullRequest,
    flux: float,
    rate_total: float,
    n_target_int: int,
    n_draw: int,
    energy_scale: float,
    angle_scale: float,
    energy_acceptance: float,
    angle_acceptance: float,
    eps_mu: float,
    n_below_mass: int,
    energy_clip_info: dict,
    angle_clip_info: dict,
    elapsed_s: float,
) -> dict:
    return {
        "created_at": time.time(),
        "tipo": "full",
        "bundle_energy": bundle_energy.name,
        "bundle_angle": bundle_angle.name,
        "trial_dir_energy": str(bundle_energy.trial_dir),
        "trial_dir_angle": str(bundle_angle.trial_dir),
        "bx_uT": float(req.bx),
        "bz_uT": float(req.bz),
        "altura_m": float(req.altura),
        "flux_part_m2_s": float(flux),
        "duration_s": float(SIM_DURATION_SECONDS),
        "area_m2": float(AREA_M2),
        "rate_total_part_m2_s": float(rate_total),
        "N_target": int(n_target_int),
        "N_draw": int(n_draw),
        "energy_scale": float(energy_scale),
        "angle_scale": float(angle_scale),
        "energy_acceptance": float(energy_acceptance),
        "angle_acceptance": float(angle_acceptance),
        "E_Y_MIN": float(E_Y_MIN),
        "E_Y_MAX": float(E_Y_MAX),
        "eps_mu": float(eps_mu),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "phi_distribution": "uniform_deg_[0,360)",
        "muon_mass_GeV": float(MUON_MASS_GEV),
        "pz_sign_convention": float(PZ_SIGN),
        "n_energy_below_muon_mass": int(n_below_mass),
        "clip_info_energy": energy_clip_info,
        "clip_info_angle": angle_clip_info,
        "simulation_time_s": float(elapsed_s),
    }


def sample_energy_events(
    bundle: ModelBundle,
    altura: float,
    bx: float,
    bz: float,
    n_draw: int,
    n_target_int: int,
) -> tuple[np.ndarray, float, float, dict, torch.Tensor]:
    if "energy_log" not in bundle.stats:
        raise RuntimeError(
            "stats del modelo de energía deben incluir 'energy_log' (min/max de logE)."
        )

    _, clip_info, context_tensor = _build_context_tensor(bundle, altura, bx, bz)

    y_samples, acceptance = sample_truncated_single_context_robust(
        bundle.flow,
        c1=context_tensor,
        n=n_draw,
        y_min=E_Y_MIN,
        y_max=E_Y_MAX,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    scale = (n_target_int * acceptance / n_draw) if n_draw > 0 else 1.0
    energy_samples = mm_to_energy(y_samples, bundle.stats)
    energy_samples = np.asarray(energy_samples, dtype=np.float64).reshape(-1)

    return energy_samples, scale, acceptance, clip_info, context_tensor


def sample_angle_events(
    bundle: ModelBundle,
    altura: float,
    bx: float,
    bz: float,
    n_draw: int,
    n_target_int: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict, torch.Tensor, float]:
    eps_mu = get_angle_eps_mu(bundle.stats, default=ANGLE_EPS_DEFAULT)
    try:
        if isinstance(bundle.stats.get("angle_target"), dict):
            eps_mu = float(bundle.stats["angle_target"].get("eps_clip", eps_mu))
    except Exception:
        pass

    _, clip_info, context_tensor = _build_context_tensor(bundle, altura, bx, bz)

    mu_samples, acceptance = sample_truncated_single_context_robust(
        bundle.flow,
        c1=context_tensor,
        n=n_draw,
        y_min=eps_mu,
        y_max=1.0 - eps_mu,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    theta_deg = mu_to_theta_deg(mu_samples).reshape(-1)
    scale = (n_target_int / n_draw) if n_draw > 0 else 1.0

    return (
        theta_deg,
        mu_samples.reshape(-1),
        scale,
        acceptance,
        clip_info,
        context_tensor,
        eps_mu,
    )


def simulate_energy_job(
    bundle: ModelBundle,
    req: SimRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    if "energy_log" not in bundle.stats:
        raise RuntimeError(
            "stats del modelo de energía deben incluir 'energy_log' (min/max de logE)."
        )

    flux, _, n_target_int, n_draw = _compute_run_counts(req.altura)

    _, clip_info, context_tensor = _build_context_tensor(
        bundle,
        req.altura,
        req.bx,
        req.bz,
    )

    y_samples, acceptance = sample_truncated_single_context_robust(
        bundle.flow,
        c1=context_tensor,
        n=n_draw,
        y_min=E_Y_MIN,
        y_max=E_Y_MAX,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    scale = (n_target_int * acceptance / n_draw) if n_draw > 0 else 1.0
    energy_samples = mm_to_energy(y_samples, bundle.stats)
    energy_samples = np.asarray(energy_samples, dtype=np.float64).reshape(-1)

    out_png = run_dir / "image.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"

    elapsed_s = time.perf_counter() - request_start_perf
    subtitle = (
        f"model={bundle.name} | N_draw={n_draw} | "
        f"T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"
    )

    save_energy_spectrum_plot(
        out_png,
        energy_samples,
        scale,
        "Energy Spectrum (CNF)",
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
    scale = (n_target_int / n_draw) if n_draw > 0 else 1.0

    eps_mu = get_angle_eps_mu(bundle.stats, default=ANGLE_EPS_DEFAULT)
    try:
        if isinstance(bundle.stats.get("angle_target"), dict):
            eps_mu = float(bundle.stats["angle_target"].get("eps_clip", eps_mu))
    except Exception:
        pass

    _, clip_info, context_tensor = _build_context_tensor(
        bundle,
        req.altura,
        req.bx,
        req.bz,
    )

    mu_samples, acceptance = sample_truncated_single_context_robust(
        bundle.flow,
        c1=context_tensor,
        n=n_draw,
        y_min=eps_mu,
        y_max=1.0 - eps_mu,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    theta_deg = mu_to_theta_deg(mu_samples)

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
        f"model={bundle.name} | N_draw={n_draw} | "
        f"T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"
    )

    save_angle_spectrum_plot(
        out_png=out_png,
        theta_deg=theta_plot,
        scale=scale,
        rate_total=rate_total,
        flow=bundle.flow,
        context_vec=context_tensor,
        eps_mu=eps_mu,
        title="Angular spectrum (CNF)",
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
        eps_mu=eps_mu,
        clip_info=clip_info,
        elapsed_s=elapsed_s,
    )

    _write_meta_json(out_meta, meta)

    write_angle_csv(out_csv, theta_plot, meta)
    write_angle_shw(out_shw, theta_plot, meta)
    make_zip(out_zip, [out_shw])

    return meta


def simulate_full_job(
    bundle_energy: ModelBundle,
    bundle_angle: ModelBundle,
    req: SimFullRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    flux, _, n_target_int, n_draw = _compute_run_counts(req.altura)

    energy_samples, energy_scale, energy_acceptance, energy_clip_info, _ = sample_energy_events(
        bundle_energy,
        req.altura,
        req.bx,
        req.bz,
        n_draw,
        n_target_int,
    )

    (
        theta_deg,
        _,
        angle_scale,
        angle_acceptance,
        angle_clip_info,
        angle_context_tensor,
        eps_mu,
    ) = sample_angle_events(
        bundle_angle,
        req.altura,
        req.bx,
        req.bz,
        n_draw,
        n_target_int,
    )

    if len(energy_samples) != len(theta_deg):
        raise RuntimeError("La cantidad de energías y ángulos no coincide.")

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
        f"model=energy | N_draw={n_draw} | "
        f"T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"
    )
    subtitle_angle = (
        f"model=angle | N_draw={n_draw} | "
        f"T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"
    )

    save_energy_spectrum_plot(
        out_png=out_energy_png,
        E=energy_samples,
        scale=energy_scale,
        title="Energy Spectrum (CNF)",
        subtitle=subtitle_energy,
        stats=bundle_energy.stats,
    )

    save_angle_spectrum_plot(
        out_png=out_angle_png,
        theta_deg=theta_plot,
        scale=angle_scale,
        rate_total=rate_total,
        flow=bundle_angle.flow,
        context_vec=angle_context_tensor,
        eps_mu=eps_mu,
        title="Angular spectrum (CNF)",
        subtitle=subtitle_angle,
    )

    n_below_mass = int(np.sum(energy_samples < MUON_MASS_GEV))

    meta = _build_full_meta(
        bundle_energy=bundle_energy,
        bundle_angle=bundle_angle,
        req=req,
        flux=flux,
        rate_total=rate_total,
        n_target_int=n_target_int,
        n_draw=n_draw,
        energy_scale=energy_scale,
        angle_scale=angle_scale,
        energy_acceptance=energy_acceptance,
        angle_acceptance=angle_acceptance,
        eps_mu=eps_mu,
        n_below_mass=n_below_mass,
        energy_clip_info=energy_clip_info,
        angle_clip_info=angle_clip_info,
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


