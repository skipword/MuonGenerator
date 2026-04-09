from pydantic import BaseModel


class SimRequest(BaseModel):
    bx: float
    bz: float
    altura: float
    modelo: str


class SimFullRequest(BaseModel):
    bx: float
    bz: float
    altura: float


class CityLookupRequest(BaseModel):
    city: str
    country: str = ""


class FieldFromCoordsRequest(BaseModel):
    lat: float
    lon: float
    altura: float