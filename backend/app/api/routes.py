import shutil
import time
import uuid
import requests
import traceback

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
    FieldFromCoordsRequest,
    SimFullRequest,
    SimRequest,
)

from ..services.geo import compute_bfield_from_coords, geocode_city
from ..services.jobs import (
    simulate_angle_job,
    simulate_energy_job,
    simulate_full_job,
)
from ..services.storage import cleanup_old_runs

router = APIRouter()


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

@router.post("/resolve-city")
def resolve_city(req: CityLookupRequest):
    try:
        return geocode_city(req.city, req.country)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="No se pudo consultar el servicio de geocodificación.",
        )


@router.post("/compute-bfield")
def compute_bfield(req: FieldFromCoordsRequest):
    try:
        return compute_bfield_from_coords(req.lat, req.lon, req.altura)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo calcular bx/bz: {repr(e)}",
        )

@router.post("/simulate")
async def simulate(req: SimRequest, request: Request):
    request_start_perf = time.perf_counter()

    model_key = (req.modelo or "").strip().lower()
    if model_key not in request.app.state.models:
        raise HTTPException(
            status_code=400,
            detail=f"modelo inválido: '{req.modelo}'. Usa: {list(request.app.state.models.keys())}",
        )

    bundle = request.app.state.models[model_key]

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old_runs(RUNS_DIR, RUN_TTL_SECONDS)

    async with request.app.state.sim_sema:
        try:
            if model_key == "energy":
                meta = await run_in_threadpool(
                    simulate_energy_job,
                    bundle,
                    req,
                    run_dir,
                    request_start_perf,
                )
            elif model_key == "angle":
                meta = await run_in_threadpool(
                    simulate_angle_job,
                    bundle,
                    req,
                    run_dir,
                    request_start_perf,
                )
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


@router.post("/simulate-full")
async def simulate_full(req: SimFullRequest, request: Request):
    request_start_perf = time.perf_counter()

    bundle_energy = request.app.state.models["energy"]
    bundle_angle = request.app.state.models["angle"]

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old_runs(RUNS_DIR, RUN_TTL_SECONDS)

    async with request.app.state.sim_sema:
        try:
            meta = await run_in_threadpool(
                simulate_full_job,
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
            "Simulation done. "
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


@router.get("/result/{run_id}/image.png")
def get_result_image(run_id: str):
    img_path = RUNS_DIR / run_id / "image.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(img_path, media_type="image/png", filename="image.png")


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