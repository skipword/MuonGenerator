import os
import json
import time
import uuid
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple
from contextlib import asynccontextmanager

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from nflows import transforms, distributions, flows

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

# ============================================================
# Paths
# ============================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Settings
# ============================================================
CNF_DEVICE = os.getenv("CNF_DEVICE", "cpu").lower()
DEVICE = torch.device("cuda" if (CNF_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")

MAX_CONCURRENT_SIMS = int(os.getenv("MAX_CONCURRENT_SIMS", "1"))
RUN_TTL_SECONDS = int(os.getenv("RUN_TTL_SECONDS", "3600"))

E_Y_MIN = float(os.getenv("ENERGY_Y_MIN", "0.0"))
E_Y_MAX = float(os.getenv("ENERGY_Y_MAX", "0.92"))

ANGLE_EPS_DEFAULT = float(os.getenv("ANGLE_EPS", "1e-6"))

SAMPLE_BATCH = int(os.getenv("CNF_SAMPLE_BATCH", "2000"))
OVERSAMPLE_FACTOR = int(os.getenv("CNF_OVERSAMPLE_FACTOR", "6"))
MAX_LOOPS = int(os.getenv("CNF_MAX_LOOPS", "200000"))

FLUX_A = float(os.getenv("FLUX_A", "101.123"))
FLUX_B = float(os.getenv("FLUX_B", "0.000210808"))

SIM_DURATION_SECONDS = float(os.getenv("SIM_DURATION_SECONDS", "3600"))
AREA_M2 = float(os.getenv("AREA_M2", "1.0"))

CAP_EVENTS = int(os.getenv("CAP_EVENTS", "10000000"))

# Bins para energía
BINS_E = np.logspace(-2, 4, 50)  # GeV

# Bins para ángulo
THETA_MIN_DEG = float(os.getenv("THETA_MIN_DEG", "0.0"))
THETA_MAX_DEG = float(os.getenv("THETA_MAX_DEG", "90.0"))
THETA_N_BINS  = int(os.getenv("THETA_N_BINS", "100"))
THETA_BINS = np.linspace(THETA_MIN_DEG, THETA_MAX_DEG, THETA_N_BINS)
THETA_BIN_WIDTH = float(THETA_BINS[1] - THETA_BINS[0])

CONTEXT_COLS = ["h", "bx", "bz"]

# Nueva parte: masa del muón + azimut + convención de pz
MUON_MASS_GEV = float(os.getenv("MUON_MASS_GEV", "0.1056583755"))
AZIMUTH_MIN_DEG = 0.0
AZIMUTH_MAX_DEG = 360.0
PZ_SIGN = float(os.getenv("PZ_SIGN", "-1.0"))  # -1 => muón descendente si +z apunta hacia arriba

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# ============================================================
# Registry
# ============================================================
MODEL_REGISTRY = {
    "energy": MODELS_DIR / "energy" / "trial_15",
    "angle":  MODELS_DIR / "angle"  / "trial_12",
}

# ============================================================
# Schemas
# ============================================================
class SimRequest(BaseModel):
    bx: float
    bz: float
    altura: float
    modelo: str  # "energy" | "angle"

class SimFullRequest(BaseModel):
    bx: float
    bz: float
    altura: float

# ============================================================
# Utilidades JSON
# ============================================================
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

# ============================================================
# MinMax
# ============================================================
def _minmax(v: float, vmin: float, vmax: float) -> float:
    return (v - vmin) / (vmax - vmin)

def ctx_to_mm(altura: float, bx: float, bz: float, stats: dict, clip: bool = True) -> Tuple[np.ndarray, dict]:
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

# ============================================================
# Energy conversions
# ============================================================
def mm_to_energy(y: np.ndarray, stats: dict) -> np.ndarray:
    e_min = float(stats["energy_log"]["min"])
    e_max = float(stats["energy_log"]["max"])
    z = y * (e_max - e_min) + e_min
    return np.exp(z)

# ============================================================
# Angle conversions
# ============================================================
def mu_to_theta_deg(mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu.astype(np.float64), 0.0, 1.0)
    return np.rad2deg(np.arccos(mu)).astype(np.float32)

def pdf_theta_deg_from_pdf_mu(grid_theta_deg: np.ndarray, pdf_mu_vals: np.ndarray) -> np.ndarray:
    th_rad = np.deg2rad(grid_theta_deg.astype(np.float64))
    return (pdf_mu_vals * np.sin(th_rad) * (np.pi / 180.0)).astype(np.float64)

# ============================================================
# Muestreo truncado
# ============================================================
@torch.no_grad()
def sample_truncated_single_context_robust(
    flow: flows.Flow,
    c1: torch.Tensor,
    n: int,
    y_min: float,
    y_max: float,
    batch: int,
    oversample_factor: int,
    max_loops: int,
    shuffle_accept: bool = True,
) -> Tuple[np.ndarray, float]:
    flow.eval()

    if c1.dim() == 1:
        c1 = c1.view(1, -1)
    if c1.size(0) != 1:
        c1 = c1[:1]

    out_cpu = []
    need = int(n)
    drawn = 0
    kept = 0
    loops = 0

    while need > 0:
        loops += 1
        if loops > max_loops:
            raise RuntimeError(
                f"Demasiados loops (>{max_loops}). drawn={drawn}, kept={kept}, need={need}. "
                f"Truncación muy estricta o contexto fuera de rango."
            )

        m = max(batch, min(batch * oversample_factor, need * oversample_factor))
        m = int(m)

        y = flow.sample(m, context=c1)

        if y.dim() == 3:
            y = y.squeeze(0)
        y = y.reshape(-1)

        drawn += y.numel()

        mask = (y >= y_min) & (y <= y_max)
        y = y[mask]
        if y.numel() == 0:
            continue

        if shuffle_accept and y.numel() > 1:
            perm = torch.randperm(y.numel(), device=y.device)
            y = y[perm]

        take = min(need, y.numel())
        out_cpu.append(y[:take].detach().cpu())
        kept += take
        need -= take

    acc = kept / drawn if drawn > 0 else float("nan")
    return torch.cat(out_cpu, dim=0).numpy(), float(acc)

# ============================================================
# Loader de modelos
# ============================================================
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
        f"Crea stats_angle.json (o stats_minmax.json) con h,bx,bz (y para energía: energy_log)."
    )

def build_flow_from_config(trial_dir: Path, context_features: int, device: torch.device) -> Tuple[flows.Flow, dict, dict]:
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config.json en {trial_dir}")

    cfg = load_json(config_path)

    hidden_features = int(cfg["hidden_features"])
    num_layers     = int(cfg["num_layers"])
    num_bins       = int(cfg["num_bins"])
    tail_bound     = float(cfg["tail_bound"])

    min_bin_width  = float(cfg.get("min_bin_width", 1e-3))
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

# ============================================================
# Cleanup runs
# ============================================================
def cleanup_old_runs(runs_dir: Path, ttl_seconds: int):
    if ttl_seconds <= 0:
        return
    now = time.time()
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        try:
            created = d.stat().st_mtime
            if meta.exists():
                with open(meta, "r", encoding="utf-8") as f:
                    mj = json.load(f)
                created = float(mj.get("created_at", created))
            if (now - created) > ttl_seconds:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

# ============================================================
# Plotters
# ============================================================
def save_energy_spectrum_plot(out_png: Path, E: np.ndarray, scale: float, title: str, subtitle: str, stats: dict):
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
        0.02, 0.02, subtitle,
        transform=plt.gca().transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
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
            g = grid_mu[i:i+chunk].astype(np.float32)
            g_t = torch.from_numpy(g).unsqueeze(1).to(device)
            ctx = context_vec.repeat(g_t.size(0), 1)
            logp = flow.log_prob(g_t, context=ctx).detach().cpu().numpy()
            pdf_list.append(np.exp(logp))
    pdf_mu = np.concatenate(pdf_list, axis=0).reshape(-1)
    return np.clip(pdf_mu, 1e-30, None)

def save_angle_spectrum_plot(
    out_png: Path,
    theta_deg: np.ndarray,
    scale: float,
    rate_total: float,
    flow: flows.Flow,
    context_vec: torch.Tensor,
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
        0.02, 0.02, subtitle,
        transform=plt.gca().transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
    )
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ============================================================
# CSV / SHW viejos
# ============================================================
def write_energy_csv(out_csv: Path, energies: np.ndarray, meta: dict):
    import csv
    E = np.asarray(energies, dtype=np.float64).reshape(-1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (ENERGY)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow(["energy_GeV"])

        for e in E:
            writer.writerow([f"{float(e):.10e}"])

def write_angle_csv(out_csv: Path, theta_deg: np.ndarray, meta: dict):
    import csv
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (ANGLE)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow(["theta_deg"])

        for t in th:
            s = f"{float(t):.8f}".replace(".", ",")
            writer.writerow([s])

def write_energy_shw(out_shw: Path, energies: np.ndarray, meta: dict):
    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: energy_GeV\n")
        for e in energies:
            f.write(f"{float(e):.10e}\n")

def write_angle_shw(out_shw: Path, theta_deg: np.ndarray, meta: dict):
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: theta_deg\n")
        for t in th:
            f.write(f"{float(t):.8f}\n")

# ============================================================
# CSV / SHW nuevos completos
# ============================================================
def write_full_csv(
    out_csv: Path,
    energy_GeV: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    p_GeV_c: np.ndarray,
    px_GeV_c: np.ndarray,
    py_GeV_c: np.ndarray,
    pz_GeV_c: np.ndarray,
    meta: dict,
):
    import csv

    E = np.asarray(energy_GeV, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    ph = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    p = np.asarray(p_GeV_c, dtype=np.float64).reshape(-1)
    px = np.asarray(px_GeV_c, dtype=np.float64).reshape(-1)
    py = np.asarray(py_GeV_c, dtype=np.float64).reshape(-1)
    pz = np.asarray(pz_GeV_c, dtype=np.float64).reshape(-1)

    n = len(E)
    if not (len(th) == len(ph) == len(p) == len(px) == len(py) == len(pz) == n):
        raise ValueError("Las columnas del CSV completo no tienen la misma longitud.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (FULL EVENT SAMPLE)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "energy_GeV",
            "theta_deg",
            "phi_deg",
            "p_GeV_c",
            "px_GeV_c",
            "py_GeV_c",
            "pz_GeV_c",
        ])

        for i in range(n):
            writer.writerow([
                i,
                f"{E[i]:.10e}",
                f"{th[i]:.8f}",
                f"{ph[i]:.8f}",
                f"{p[i]:.10e}",
                f"{px[i]:.10e}",
                f"{py[i]:.10e}",
                f"{pz[i]:.10e}",
            ])

def write_full_shw(
    out_shw: Path,
    energy_GeV: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    p_GeV_c: np.ndarray,
    px_GeV_c: np.ndarray,
    py_GeV_c: np.ndarray,
    pz_GeV_c: np.ndarray,
    meta: dict,
):
    E = np.asarray(energy_GeV, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    ph = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    p = np.asarray(p_GeV_c, dtype=np.float64).reshape(-1)
    px = np.asarray(px_GeV_c, dtype=np.float64).reshape(-1)
    py = np.asarray(py_GeV_c, dtype=np.float64).reshape(-1)
    pz = np.asarray(pz_GeV_c, dtype=np.float64).reshape(-1)

    n = len(E)
    if not (len(th) == len(ph) == len(p) == len(px) == len(py) == len(pz) == n):
        raise ValueError("Las columnas del SHW completo no tienen la misma longitud.")

    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: event_id energy_GeV theta_deg phi_deg p_GeV_c px_GeV_c py_GeV_c pz_GeV_c\n")
        for i in range(n):
            f.write(
                f"{i} "
                f"{E[i]:.10e} "
                f"{th[i]:.8f} "
                f"{ph[i]:.8f} "
                f"{p[i]:.10e} "
                f"{px[i]:.10e} "
                f"{py[i]:.10e} "
                f"{pz[i]:.10e}\n"
            )

def make_zip(out_zip: Path, files: list[Path]):
    import zipfile
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)

# ============================================================
# Flujo y conteos
# ============================================================
def flux_per_m2_per_s(h: float) -> float:
    return FLUX_A * math.exp(FLUX_B * float(h))

def target_counts(h: float, duration_s: float, area_m2: float) -> float:
    return flux_per_m2_per_s(h) * float(duration_s) * float(area_m2)

# ============================================================
# Nueva parte: azimut y momento
# ============================================================
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

# ============================================================
# Model bundle
# ============================================================
class ModelBundle:
    def __init__(self, name: str, trial_dir: Path, flow: flows.Flow, cfg: dict, stats: dict, device: torch.device):
        self.name = name
        self.trial_dir = trial_dir
        self.flow = flow
        self.cfg = cfg
        self.stats = stats
        self.device = device

def load_all_models() -> Dict[str, ModelBundle]:
    bundles: Dict[str, ModelBundle] = {}
    for name, trial_dir in MODEL_REGISTRY.items():
        flow, cfg, stats = build_flow_from_config(trial_dir, context_features=len(CONTEXT_COLS), device=DEVICE)
        bundles[name] = ModelBundle(name, trial_dir, flow, cfg, stats, DEVICE)
    return bundles

# ============================================================
# Helpers nuevos para muestrear energía y ángulo
# ============================================================
def _sample_energy_events(bundle: ModelBundle, req_altura: float, req_bx: float, req_bz: float, n_draw: int, n_target_int: int):
    if "energy_log" not in bundle.stats:
        raise RuntimeError("stats del modelo de energía deben incluir 'energy_log' (min/max de logE).")

    c_mm, clip_info = ctx_to_mm(req_altura, req_bx, req_bz, bundle.stats, clip=True)
    c1 = torch.tensor(c_mm, dtype=torch.float32, device=bundle.device).view(1, -1)

    y_samp, acc = sample_truncated_single_context_robust(
        bundle.flow,
        c1=c1,
        n=n_draw,
        y_min=E_Y_MIN,
        y_max=E_Y_MAX,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    scale = (n_target_int * acc / n_draw) if n_draw > 0 else 1.0
    E_samp = mm_to_energy(y_samp, bundle.stats)
    E_samp = np.asarray(E_samp, dtype=np.float64).reshape(-1)
    return E_samp, scale, acc, clip_info, c1

def _sample_angle_events(bundle: ModelBundle, req_altura: float, req_bx: float, req_bz: float, n_draw: int, n_target_int: int):
    eps_mu = get_angle_eps_mu(bundle.stats, default=ANGLE_EPS_DEFAULT)
    try:
        if isinstance(bundle.stats.get("angle_target", None), dict):
            eps_mu = float(bundle.stats["angle_target"].get("eps_clip", eps_mu))
    except Exception:
        pass

    c_mm, clip_info = ctx_to_mm(req_altura, req_bx, req_bz, bundle.stats, clip=True)
    c1 = torch.tensor(c_mm, dtype=torch.float32, device=bundle.device).view(1, -1)

    mu_samp, acc = sample_truncated_single_context_robust(
        bundle.flow,
        c1=c1,
        n=n_draw,
        y_min=eps_mu,
        y_max=1.0 - eps_mu,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    theta_deg = mu_to_theta_deg(mu_samp).reshape(-1)
    scale = (n_target_int / n_draw) if n_draw > 0 else 1.0
    return theta_deg, mu_samp.reshape(-1), scale, acc, clip_info, c1, eps_mu

# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_models()
    import asyncio
    app.state.sim_sema = asyncio.Semaphore(MAX_CONCURRENT_SIMS)
    yield

app = FastAPI(title="Backend Simulador Muones (CNF energy + angle)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Health
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_models": list(app.state.models.keys()),
        "settings": {
            "E_Y_MIN": E_Y_MIN,
            "E_Y_MAX": E_Y_MAX,
            "CAP_EVENTS": CAP_EVENTS,
            "SIM_DURATION_SECONDS": SIM_DURATION_SECONDS,
            "AREA_M2": AREA_M2,
            "MAX_CONCURRENT_SIMS": MAX_CONCURRENT_SIMS,
            "RUN_TTL_SECONDS": RUN_TTL_SECONDS,
            "FLUX_A": FLUX_A,
            "FLUX_B": FLUX_B,
            "THETA_MIN_DEG": THETA_MIN_DEG,
            "THETA_MAX_DEG": THETA_MAX_DEG,
            "THETA_N_BINS": THETA_N_BINS,
            "MUON_MASS_GEV": MUON_MASS_GEV,
            "PZ_SIGN": PZ_SIGN,
        }
    }

# ============================================================
# Jobs viejos
# ============================================================
def _simulate_energy_job(bundle: ModelBundle, req: SimRequest, run_dir: Path, request_start_perf: float) -> dict:
    if "energy_log" not in bundle.stats:
        raise RuntimeError("stats del modelo de energía deben incluir 'energy_log' (min/max de logE).")

    flux = flux_per_m2_per_s(req.altura)
    N_tgt = target_counts(req.altura, SIM_DURATION_SECONDS, AREA_M2)
    N_target_int = int(max(0, round(N_tgt)))

    N_draw = int(min(N_target_int, CAP_EVENTS))
    if N_draw <= 0:
        raise RuntimeError("N_draw <= 0. Revisa altura y fórmula de flujo.")

    c_mm, clip_info = ctx_to_mm(req.altura, req.bx, req.bz, bundle.stats, clip=True)
    c1 = torch.tensor(c_mm, dtype=torch.float32, device=bundle.device).view(1, -1)

    y_samp, acc = sample_truncated_single_context_robust(
        bundle.flow,
        c1=c1,
        n=N_draw,
        y_min=E_Y_MIN,
        y_max=E_Y_MAX,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )
    scale = (N_target_int * acc / N_draw) if N_draw > 0 else 1.0
    E_samp = mm_to_energy(y_samp, bundle.stats)
    E_samp = np.asarray(E_samp, dtype=np.float64).reshape(-1)

    out_png = run_dir / "image.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"
    elapsed_s = time.perf_counter() - request_start_perf

    subtitle = f"model={bundle.name} | N_draw={N_draw} | T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"

    save_energy_spectrum_plot(
        out_png, E_samp, scale,
        "Energy Spectrum (CNF)",
        subtitle,
        stats=bundle.stats
    )

    meta = {
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
        "N_target": int(N_target_int),
        "N_draw": int(N_draw),
        "scale": float(scale),
        "acceptance": float(acc),
        "E_Y_MIN": float(E_Y_MIN),
        "E_Y_MAX": float(E_Y_MAX),
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_energy_csv(out_csv, E_samp, meta)
    write_energy_shw(out_shw, E_samp, meta)
    make_zip(out_zip, [out_shw])

    return meta

def _simulate_angle_job(bundle: ModelBundle, req: SimRequest, run_dir: Path, request_start_perf: float) -> dict:
    flux = flux_per_m2_per_s(req.altura)
    N_tgt = target_counts(req.altura, SIM_DURATION_SECONDS, AREA_M2)
    N_target_int = int(max(0, round(N_tgt)))

    N_draw = int(min(N_target_int, CAP_EVENTS))
    if N_draw <= 0:
        raise RuntimeError("N_draw <= 0. Revisa altura y fórmula de flujo.")
    scale = (N_target_int / N_draw) if N_draw > 0 else 1.0

    eps_mu = get_angle_eps_mu(bundle.stats, default=ANGLE_EPS_DEFAULT)
    try:
        if isinstance(bundle.stats.get("angle_target", None), dict):
            eps_mu = float(bundle.stats["angle_target"].get("eps_clip", eps_mu))
    except Exception:
        pass

    c_mm, clip_info = ctx_to_mm(req.altura, req.bx, req.bz, bundle.stats, clip=True)
    c1 = torch.tensor(c_mm, dtype=torch.float32, device=bundle.device).view(1, -1)

    mu_samp, acc = sample_truncated_single_context_robust(
        bundle.flow,
        c1=c1,
        n=N_draw,
        y_min=eps_mu,
        y_max=1.0 - eps_mu,
        batch=SAMPLE_BATCH,
        oversample_factor=OVERSAMPLE_FACTOR,
        max_loops=MAX_LOOPS,
        shuffle_accept=True,
    )

    theta_deg = mu_to_theta_deg(mu_samp)

    mask_plot = (theta_deg >= THETA_MIN_DEG) & (theta_deg <= THETA_MAX_DEG)
    theta_plot = theta_deg[mask_plot]

    out_png = run_dir / "image.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"
    elapsed_s = time.perf_counter() - request_start_perf
    rate_total = (N_target_int / SIM_DURATION_SECONDS)

    subtitle = f"model={bundle.name} | N_draw={N_draw} | T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"

    save_angle_spectrum_plot(
        out_png=out_png,
        theta_deg=theta_plot,
        scale=scale,
        rate_total=rate_total,
        flow=bundle.flow,
        context_vec=c1,
        eps_mu=eps_mu,
        title="Angular spectrum (CNF)",
        subtitle=subtitle,
    )

    meta = {
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
        "N_target": int(N_target_int),
        "N_draw": int(N_draw),
        "scale": float(scale),
        "acceptance": float(acc),
        "eps_mu": float(eps_mu),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "clip_info": clip_info,
        "simulation_time_s": float(elapsed_s),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_angle_csv(out_csv, theta_plot, meta)
    write_angle_shw(out_shw, theta_plot, meta)
    make_zip(out_zip, [out_shw])
    return meta

# ============================================================
# Job nuevo completo
# ============================================================
def _simulate_full_job(
    bundle_energy: ModelBundle,
    bundle_angle: ModelBundle,
    req: SimFullRequest,
    run_dir: Path,
    request_start_perf: float,
) -> dict:
    flux = flux_per_m2_per_s(req.altura)
    N_tgt = target_counts(req.altura, SIM_DURATION_SECONDS, AREA_M2)
    N_target_int = int(max(0, round(N_tgt)))

    N_draw = int(min(N_target_int, CAP_EVENTS))
    if N_draw <= 0:
        raise RuntimeError("N_draw <= 0. Revisa altura y fórmula de flujo.")

    E_samp, energy_scale, energy_acc, energy_clip_info, _ = _sample_energy_events(
        bundle_energy, req.altura, req.bx, req.bz, N_draw, N_target_int
    )

    theta_deg, _, angle_scale, angle_acc, angle_clip_info, c1_angle, eps_mu = _sample_angle_events(
        bundle_angle, req.altura, req.bx, req.bz, N_draw, N_target_int
    )

    if len(E_samp) != len(theta_deg):
        raise RuntimeError("La cantidad de energías y ángulos no coincide.")

    phi_deg = sample_phi_deg_uniform(N_draw)
    p_GeV_c = momentum_from_total_energy_GeV(E_samp)
    px_GeV_c, py_GeV_c, pz_GeV_c = momentum_components_from_angles(p_GeV_c, theta_deg, phi_deg)

    mask_plot = (theta_deg >= THETA_MIN_DEG) & (theta_deg <= THETA_MAX_DEG)
    theta_plot = theta_deg[mask_plot]

    out_energy_png = run_dir / "energy.png"
    out_angle_png = run_dir / "angle.png"
    out_csv = run_dir / "results.csv"
    out_meta = run_dir / "meta.json"
    out_shw = run_dir / "results.shw"
    out_zip = run_dir / "results_shw.zip"

    elapsed_s = time.perf_counter() - request_start_perf
    rate_total = (N_target_int / SIM_DURATION_SECONDS)

    subtitle_energy = f"model=energy | N_draw={N_draw} | T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"
    subtitle_angle = f"model=angle | N_draw={N_draw} | T={SIM_DURATION_SECONDS:g}s | sim_time={elapsed_s:.2f}s"

    save_energy_spectrum_plot(
        out_png=out_energy_png,
        E=E_samp,
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
        context_vec=c1_angle,
        eps_mu=eps_mu,
        title="Angular spectrum (CNF)",
        subtitle=subtitle_angle,
    )

    n_below_mass = int(np.sum(E_samp < MUON_MASS_GEV))

    meta = {
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
        "N_target": int(N_target_int),
        "N_draw": int(N_draw),
        "energy_scale": float(energy_scale),
        "angle_scale": float(angle_scale),
        "energy_acceptance": float(energy_acc),
        "angle_acceptance": float(angle_acc),
        "E_Y_MIN": float(E_Y_MIN),
        "E_Y_MAX": float(E_Y_MAX),
        "eps_mu": float(eps_mu),
        "theta_plot_range_deg": [float(THETA_MIN_DEG), float(THETA_MAX_DEG)],
        "phi_distribution": "uniform_deg_[0,360)",
        "muon_mass_GeV": float(MUON_MASS_GEV),
        "pz_sign_convention": float(PZ_SIGN),
        "n_energy_below_muon_mass": n_below_mass,
        "clip_info_energy": energy_clip_info,
        "clip_info_angle": angle_clip_info,
        "simulation_time_s": float(elapsed_s),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_full_csv(
        out_csv=out_csv,
        energy_GeV=E_samp,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        p_GeV_c=p_GeV_c,
        px_GeV_c=px_GeV_c,
        py_GeV_c=py_GeV_c,
        pz_GeV_c=pz_GeV_c,
        meta=meta,
    )

    write_full_shw(
        out_shw=out_shw,
        energy_GeV=E_samp,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        p_GeV_c=p_GeV_c,
        px_GeV_c=px_GeV_c,
        py_GeV_c=py_GeV_c,
        pz_GeV_c=pz_GeV_c,
        meta=meta,
    )
    make_zip(out_zip, [out_shw])

    return meta

# ============================================================
# API
# ============================================================
@app.post("/simulate")
async def simulate(req: SimRequest, request: Request):
    request_start_perf = time.perf_counter()

    model_key = (req.modelo or "").strip().lower()
    if model_key not in app.state.models:
        raise HTTPException(
            status_code=400,
            detail=f"modelo inválido: '{req.modelo}'. Usa: {list(app.state.models.keys())}"
        )

    bundle: ModelBundle = app.state.models[model_key]

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old_runs(RUNS_DIR, RUN_TTL_SECONDS)

    async with app.state.sim_sema:
        try:
            if model_key == "energy":
                meta = await run_in_threadpool(_simulate_energy_job, bundle, req, run_dir, request_start_perf)
            elif model_key == "angle":
                meta = await run_in_threadpool(_simulate_angle_job, bundle, req, run_dir, request_start_perf)
            else:
                raise HTTPException(status_code=400, detail="modelo no soportado")
        except Exception as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))

    total_request_elapsed_s = time.perf_counter() - request_start_perf

    base = str(request.base_url).rstrip("/")
    return {
        "message": (
            f"Simulation done (modelo={req.modelo}). "
            f"N_target={meta['N_target']} | N_draw={meta['N_draw']} | "
            f"acc≈{meta['acceptance']:.3f} | time={total_request_elapsed_s:.2f}s"
        ),
        "image_url": f"{base}/result/{run_id}/image.png",
        "download_csv_url": f"{base}/download/{run_id}/results.csv",
        "download_shw_url": f"{base}/download/{run_id}/results_shw.zip",
        "run_id": run_id,
        "simulation_time_s": round(total_request_elapsed_s, 4),
    }

@app.post("/simulate-full")
async def simulate_full(req: SimFullRequest, request: Request):
    request_start_perf = time.perf_counter()

    bundle_energy: ModelBundle = app.state.models["energy"]
    bundle_angle: ModelBundle = app.state.models["angle"]

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old_runs(RUNS_DIR, RUN_TTL_SECONDS)

    async with app.state.sim_sema:
        try:
            meta = await run_in_threadpool(
                _simulate_full_job,
                bundle_energy,
                bundle_angle,
                req,
                run_dir,
                request_start_perf,
            )
        except Exception as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))

    total_request_elapsed_s = time.perf_counter() - request_start_perf
    base = str(request.base_url).rstrip("/")

    return {
        "message": (
            f"Simulation done. "
            f"N_target={meta['N_target']} | N_draw={meta['N_draw']} | "
            f"accE≈{meta['energy_acceptance']:.3f} | accA≈{meta['angle_acceptance']:.3f} | "
            f"time={total_request_elapsed_s:.2f}s"
        ),
        "image_urls": [
            f"{base}/result/{run_id}/energy.png",
            f"{base}/result/{run_id}/angle.png",
        ],
        "image_labels": [
            "Energy spectrum",
            "Angular spectrum",
        ],
        "download_csv_url": f"{base}/download/{run_id}/results.csv",
        "download_shw_url": f"{base}/download/{run_id}/results_shw.zip",
        "run_id": run_id,
        "simulation_time_s": round(total_request_elapsed_s, 4),
    }

# ============================================================
# Result images
# ============================================================
@app.get("/result/{run_id}/image.png")
def get_result_image(run_id: str):
    img_path = RUNS_DIR / run_id / "image.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="image.png")

@app.get("/result/{run_id}/energy.png")
def get_result_energy_image(run_id: str):
    img_path = RUNS_DIR / run_id / "energy.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen de energía no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="energy.png")

@app.get("/result/{run_id}/angle.png")
def get_result_angle_image(run_id: str):
    img_path = RUNS_DIR / run_id / "angle.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen angular no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="angle.png")

# ============================================================
# Downloads
# ============================================================
@app.get("/download/{run_id}/results.csv")
def download_csv(run_id: str):
    fpath = RUNS_DIR / run_id / "results.csv"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="CSV no encontrado")
    return FileResponse(fpath, media_type="text/csv", filename="results.csv")

@app.get("/download/{run_id}/results_shw.zip")
def download_shw_zip(run_id: str):
    fpath = RUNS_DIR / run_id / "results_shw.zip"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="ZIP SHW no encontrado")
    return FileResponse(fpath, media_type="application/zip", filename="results_shw.zip")