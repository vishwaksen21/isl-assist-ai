from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.dataset import router as dataset_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.predict import router as predict_router
from app.core.config import settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    app.include_router(predict_router, prefix=settings.api_prefix)
    app.include_router(feedback_router, prefix=settings.api_prefix)
    app.include_router(dataset_router, prefix=settings.api_prefix)
    return app


app = create_app()
