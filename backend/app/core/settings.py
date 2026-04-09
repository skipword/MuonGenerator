import os
from pathlib import Path

import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CNF_DEVICE = os.getenv("CNF_DEVICE", "cpu").lower()
DEVICE = torch.device(
    "cuda" if (CNF_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
)

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

BINS_E = np.logspace(-2, 4, 50)

THETA_MIN_DEG = float(os.getenv("THETA_MIN_DEG", "0.0"))
THETA_MAX_DEG = float(os.getenv("THETA_MAX_DEG", "90.0"))
THETA_N_BINS = int(os.getenv("THETA_N_BINS", "100"))
THETA_BINS = np.linspace(THETA_MIN_DEG, THETA_MAX_DEG, THETA_N_BINS)
THETA_BIN_WIDTH = float(THETA_BINS[1] - THETA_BINS[0])

CONTEXT_COLS = ["h", "bx", "bz"]

MUON_MASS_GEV = float(os.getenv("MUON_MASS_GEV", "0.1056583755"))
AZIMUTH_MIN_DEG = 0.0
AZIMUTH_MAX_DEG = 360.0
PZ_SIGN = float(os.getenv("PZ_SIGN", "1.0"))

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

MODEL_REGISTRY = {
    "energy": MODELS_DIR / "energy" / "trial_15",
    "angle": MODELS_DIR / "angle" / "trial_12",
}

GEOCODER_BASE_URL = os.getenv(
    "GEOCODER_BASE_URL",
    "https://nominatim.openstreetmap.org/search",
)

GEOCODER_USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT",
    "muon-simulator/1.0 (contact: cristianexclusivo1@gmail.com)",
)

GEOCODER_TIMEOUT_SECONDS = float(
    os.getenv("GEOCODER_TIMEOUT_SECONDS", "20")
)