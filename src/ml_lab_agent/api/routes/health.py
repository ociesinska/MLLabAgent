from __future__ import annotations

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health", status_code=200)
def health():
    return {"status": "ok"}
