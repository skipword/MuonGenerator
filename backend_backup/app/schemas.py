from pydantic import BaseModel, ConfigDict, Field


class CityLookupRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    city: str = Field(..., min_length=1)
    country: str = ""


class CityLookupResponse(BaseModel):
    query: str
    display_name: str
    lat: float
    lon: float


class FieldFromCoordsRequest(BaseModel):
    lat: float
    lon: float
    altura: float = Field(..., ge=0)


class FieldFromCoordsResponse(BaseModel):
    lat: float
    lon: float
    altura: float
    bx: float
    bz: float
    computed_at_utc: str
    model: str


class SimFullRequest(BaseModel):
    bx: float
    bz: float
    altura: float = Field(..., ge=0)


class SimFullResponse(BaseModel):
    message: str
    image_urls: list[str]
    image_labels: list[str]
    download_csv_url: str
    download_shw_url: str
    run_id: str
    simulation_time_s: float