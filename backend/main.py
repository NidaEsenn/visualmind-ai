"""
VisualMind AI — FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models.schemas import HealthResponse, StatsResponse
from routers import eval as eval_router
from routers import images as images_router
from routers import search as search_router
from services.clip_service import clip_service
from services.qdrant_service import qdrant_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    storage_path = Path(os.getenv("IMAGE_STORAGE_PATH", "./data/images"))
    thumbs_path = storage_path / "thumbs"
    storage_path.mkdir(parents=True, exist_ok=True)
    thumbs_path.mkdir(parents=True, exist_ok=True)
    logger.info("Image storage ready at %s", storage_path.resolve())

    # Ensure Qdrant collection exists (idempotent)
    qdrant_service._ensure_collection()
    logger.info("Qdrant collection '%s' ready.", qdrant_service.collection)

    yield

    # --- Shutdown ---
    logger.info("VisualMind AI shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VisualMind AI",
    description="Semantic visual search engine for designers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — configurable via CORS_ORIGINS env var (comma-separated)
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for stored images
_storage_path = Path(os.getenv("IMAGE_STORAGE_PATH", "./data/images"))
_storage_path.mkdir(parents=True, exist_ok=True)
# Static file serving: actual image/thumbnail files served at /static/images
app.mount("/static/images", StaticFiles(directory=str(_storage_path)), name="images")

# Routers — API routes under /api prefix to avoid conflict with static mount
app.include_router(images_router.router, prefix="/api/images", tags=["images"])
app.include_router(search_router.router, prefix="/api/search", tags=["search"])
app.include_router(eval_router.router, prefix="/api/eval", tags=["eval"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    qdrant_ok = False
    image_count = 0
    try:
        stats = qdrant_service.get_collection_stats()
        qdrant_ok = True
        image_count = stats.get("count", 0)
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)

    model_ok = clip_service is not None

    return HealthResponse(
        status="ok" if (qdrant_ok and model_ok) else "degraded",
        qdrant=qdrant_ok,
        model=model_ok,
        model_name=clip_service.model_name if model_ok else "unavailable",
        image_count=image_count,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=StatsResponse, tags=["meta"])
async def stats():
    try:
        col_stats = qdrant_service.get_collection_stats()
    except Exception as exc:
        logger.warning("Failed to get collection stats: %s", exc)
        col_stats = {"count": 0, "index_size_mb": 0.0}

    # Build tag distribution by scrolling all payloads
    tag_distribution: dict = {
        "layout_type": {},
        "color_mood": {},
        "industry": {},
        "complexity": {},
    }
    collection_ids: set[str] = set()

    try:
        points, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            p = point.payload or {}
            for field in ("layout_type", "color_mood", "industry", "complexity"):
                val = p.get(field) or (p.get("tags") or {}).get(field, "other")
                if val:
                    tag_distribution[field][val] = (
                        tag_distribution[field].get(val, 0) + 1
                    )
            cid = p.get("collection_id")
            if cid:
                collection_ids.add(cid)
    except Exception as exc:
        logger.warning("Could not build tag distribution: %s", exc)

    return StatsResponse(
        total_images=col_stats.get("count", 0),
        total_collections=len(collection_ids),
        tag_distribution=tag_distribution,
        index_size_mb=col_stats.get("index_size_mb", 0.0),
    )
