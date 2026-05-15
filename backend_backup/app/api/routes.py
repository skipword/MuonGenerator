import logging
import shutil
import time
import uuid

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..core.settings import (
    AREA_M2,
    CAP_EVENTS,
    DEVICE,
    E_Y_MAX,
    E_Y_MIN,
    FLUX_A,
    FLUX_B,
    MAX_CONCURRENT_SIMS,
    MUON_MASS_GEV,
    PZ_SIGN,
    RUNS_DIR,
    RUN_TTL_SECONDS,
    SIM_DURATION_SECONDS,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
    THETA_N_BINS,
)
from ..schemas import (
    CityLookupRequest,
    CityLookupResponse,
    FieldFromCoordsRequest,
    FieldFromCoordsResponse,
    SimFullRequest,
    SimFullResponse,
)
from ..services.geo import compute_bfield_from_coords, geocode_city
from ..services.jobs import simulate_full_job
from ..services.storage import cleanup_old_runs

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_models": list(request.app.state.models.keys()),
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
        },
    }


@router.post("/resolve-city", response_model=CityLookupResponse)
def resolve_city(req: CityLookupRequest) -> CityLookupResponse:
    try:
        result = geocode_city(req.city, req.country)
        return CityLookupResponse(**result)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="No se pudo encontrar la ciudad indicada.",
        )
    except requests.RequestException:
        logger.exception("Fallo al consultar el servicio de geocodificación.")
        raise HTTPException(
            status_code=502,
            detail="No se pudo consultar el servicio de geocodificación.",
        )


@router.post("/compute-bfield", response_model=FieldFromCoordsResponse)
def compute_bfield(req: FieldFromCoordsRequest) -> FieldFromCoordsResponse:
    try:
        result = compute_bfield_from_coords(req.lat, req.lon, req.altura)
        return FieldFromCoordsResponse(**result)
    except Exception:
        logger.exception("Fallo al calcular el campo geomagnético.")
        raise HTTPException(
            status_code=500,
            detail="No se pudo calcular el campo geomagnético.",
        )


@router.post("/simulate-full", response_model=SimFullResponse)
async def simulate_full(
    req: SimFullRequest,
    request: Request,
) -> SimFullResponse:
    request_start_perf = time.perf_counter()

    bundle = request.app.state.models["joint"]

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old_runs(RUNS_DIR, RUN_TTL_SECONDS)

    async with request.app.state.sim_sema:
        try:
            await run_in_threadpool(
                simulate_full_job,
                bundle,
                req,
                run_dir,
                request_start_perf,
            )
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.exception("Fallo al ejecutar la simulación completa.")
            raise HTTPException(
                status_code=500,
                detail="No se pudo completar la simulación.",
            )

    total_request_elapsed_s = time.perf_counter() - request_start_perf
    base = str(request.base_url).rstrip("/")

    return SimFullResponse(
        message="Simulación terminada.",
        image_urls=[
            f"{base}/result/{run_id}/energy.png",
            f"{base}/result/{run_id}/angle.png",
        ],
        image_labels=[
            "Espectro de energía",
            "Espectro angular",
        ],
        download_csv_url=f"{base}/download/{run_id}/results.csv",
        download_shw_url=f"{base}/download/{run_id}/results_shw.zip",
        run_id=run_id,
        simulation_time_s=round(total_request_elapsed_s, 4),
    )


@router.get("/result/{run_id}/energy.png")
def get_result_energy_image(run_id: str):
    img_path = RUNS_DIR / run_id / "energy.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen de energía no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="energy.png")


@router.get("/result/{run_id}/angle.png")
def get_result_angle_image(run_id: str):
    img_path = RUNS_DIR / run_id / "angle.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen angular no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="angle.png")


@router.get("/download/{run_id}/results.csv")
def download_csv(run_id: str):
    fpath = RUNS_DIR / run_id / "results.csv"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="CSV no encontrado")
    return FileResponse(fpath, media_type="text/csv", filename="results.csv")


@router.get("/download/{run_id}/results_shw.zip")
def download_shw_zip(run_id: str):
    fpath = RUNS_DIR / run_id / "results_shw.zip"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="ZIP SHW no encontrado")
    return FileResponse(fpath, media_type="application/zip", filename="results_shw.zip")