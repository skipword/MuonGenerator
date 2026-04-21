from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.settings import MAX_CONCURRENT_SIMS
from .services.model_loader import load_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_models()

    import asyncio
    app.state.sim_sema = asyncio.Semaphore(MAX_CONCURRENT_SIMS)

    yield


app = FastAPI(
    title="Backend Simulador Muones (CNF joint energy+angle)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)