from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.config import get_settings
from app.logging_config import configure_logging


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    logger.info("Starting %s", settings.app_name)

    yield

    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Enterprise Multimodal Document Intelligence "
        "and Autonomous Research Agent"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(query_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Return the service health status."""

    return {
        "status": "ok",
        "service": "docmind-ai",
    }
