from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import legacy, scientific, system

app = FastAPI(
    title="Colorectal Cancer Detection API",
    version="1.1.0",
    description="Inference, legacy migration, and reproducible scientific workflow API for this project.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/v1")
app.include_router(legacy.router, prefix="/api/v1")
app.include_router(scientific.router, prefix="/api/v1")