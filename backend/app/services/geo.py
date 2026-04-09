import math
from datetime import datetime, timezone

import numpy as np
import ppigrf
import requests

from ..core.settings import (
    GEOCODER_BASE_URL,
    GEOCODER_TIMEOUT_SECONDS,
    GEOCODER_USER_AGENT,
)


def _to_scalar_float(value, name: str) -> float:
    arr = np.asarray(value)

    if arr.size != 1:
        raise ValueError(
            f"{name} no es escalar. shape={arr.shape}, value={value}"
        )

    return float(arr.reshape(-1)[0])


def geocode_city(city: str, country: str = "") -> dict:
    city = (city or "").strip()
    country = (country or "").strip()

    if not city:
        raise ValueError("Debes ingresar una ciudad.")

    query = city if not country else f"{city}, {country}"

    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
    }

    headers = {
        "User-Agent": GEOCODER_USER_AGENT,
        "Accept-Language": "es",
    }

    response = requests.get(
        GEOCODER_BASE_URL,
        params=params,
        headers=headers,
        timeout=GEOCODER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    if not data:
        raise ValueError(f"No se encontró la ciudad: {query}")

    best = data[0]

    return {
        "query": query,
        "display_name": best.get("display_name", query),
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
    }


def compute_bfield_from_coords(lat: float, lon: float, altura_m: float) -> dict:
    """
    Convención del proyecto:
      bx = horizontal intensity (H)
      bz = vertical component (Z)

    ppigrf.igrf(lon, lat, h_km, date) devuelve:
      Be = este
      Bn = norte
      Bu = arriba
    """
    lat = float(lat)
    lon = float(lon)
    altura_m = float(altura_m)
    altitude_km = altura_m / 1000.0

    now_utc = datetime.now(timezone.utc)
    when_naive_utc = now_utc.replace(tzinfo=None)

    Be, Bn, Bu = ppigrf.igrf(
        lon,
        lat,
        altitude_km,
        when_naive_utc,
    )

    Be = _to_scalar_float(Be, "Be")
    Bn = _to_scalar_float(Bn, "Bn")
    Bu = _to_scalar_float(Bu, "Bu")

    x_nT = Bn
    y_nT = Be
    z_nT = -Bu
    h_nT = math.sqrt(x_nT**2 + y_nT**2)

    bx_uT = h_nT / 1000.0
    bz_uT = z_nT / 1000.0

    return {
        "lat": lat,
        "lon": lon,
        "altura": altura_m,
        "bx": bx_uT,
        "bz": bz_uT,
        "horizontal_intensity_uT": bx_uT,
        "vertical_component_uT": bz_uT,
        "computed_at_utc": now_utc.isoformat(),
        "model": "IGRF (ppigrf local)",
    }