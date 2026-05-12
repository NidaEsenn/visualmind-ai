"""
Search endpoints.

GET /search/text?q=&k=20&collection_id=&filters=   — text-to-image dense search
GET /search/similar/{image_id}?k=20                — find visually similar images
GET /search/hybrid?q=&k=20&alpha=0.7               — hybrid dense + BM25 search
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    HybridSearchResponse,
    HybridSearchResult,
    ImageTags,
    SearchResponse,
    SearchResult,
)
from services.clip_service import clip_service
from services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _payload_to_tags(payload: dict) -> ImageTags:
    tags_data = payload.get("tags") or {
        "layout_type": payload.get("layout_type", "other"),
        "color_mood": payload.get("color_mood", "minimal"),
        "ui_patterns": payload.get("ui_patterns", []),
        "industry": payload.get("industry", "other"),
        "complexity": payload.get("complexity", "medium"),
    }
    return ImageTags(**tags_data)


def _tag_boost(hits: list[dict], query: str, weight: float = 0.08) -> list[dict]:
    """Boost hits whose tag fields contain query keywords, then re-sort."""
    keywords = [w for w in query.lower().split() if len(w) > 2]
    if not keywords:
        return hits
    for hit in hits:
        payload = hit.get("payload", {})
        tag_text = " ".join(filter(None, [
            payload.get("layout_type", ""),
            payload.get("color_mood", ""),
            payload.get("industry", ""),
            payload.get("complexity", ""),
            " ".join(payload.get("ui_patterns") or []),
            payload.get("filename", ""),
        ])).lower()
        boost = sum(tag_text.count(kw) for kw in keywords) * weight
        hit["score"] = hit["score"] + boost
    return sorted(hits, key=lambda h: -h["score"])


def _hit_to_search_result(hit: dict) -> SearchResult:
    payload = hit.get("payload", {})
    return SearchResult(
        id=hit["id"],
        score=hit["score"],
        url=payload.get("url", ""),
        thumbnail_url=payload.get("thumbnail_url", ""),
        tags=_payload_to_tags(payload),
    )


def _hit_to_hybrid_result(hit: dict) -> HybridSearchResult:
    payload = hit.get("payload", {})
    return HybridSearchResult(
        id=hit["id"],
        score=hit["score"],
        url=payload.get("url", ""),
        thumbnail_url=payload.get("thumbnail_url", ""),
        tags=_payload_to_tags(payload),
        dense_score=hit.get("dense_score", hit["score"]),
        sparse_score=hit.get("sparse_score", 0.0),
    )


# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------

@router.get("/text", response_model=SearchResponse)
async def text_search(
    q: str = Query(..., min_length=1, description="Natural language query"),
    k: int = Query(default=20, ge=1, le=100),
    collection_id: Optional[str] = Query(default=None),
    filters: Optional[str] = Query(
        default=None,
        description='JSON string of filter key/value pairs, e.g. {"layout_type":"dashboard"}',
    ),
):
    t0 = time.monotonic()

    # Build filter dict
    filter_dict: Optional[dict] = None
    if collection_id:
        filter_dict = filter_dict or {}
        filter_dict["collection_id"] = collection_id
    if filters:
        try:
            extra = json.loads(filters)
            filter_dict = {**(filter_dict or {}), **extra}
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="filters must be valid JSON")

    try:
        query_embedding = clip_service.encode_text(q)
        hits = qdrant_service.search(query_embedding, k=k, filters=filter_dict)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable. Is Qdrant running?")

    hits = _tag_boost(hits, q)
    results = [_hit_to_search_result(h) for h in hits]
    elapsed_ms = (time.monotonic() - t0) * 1000

    return SearchResponse(
        results=results,
        total=len(results),
        query=q,
        query_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Similar-image search
# ---------------------------------------------------------------------------

@router.get("/similar/{image_id}", response_model=SearchResponse)
async def similar_search(
    image_id: str,
    k: int = Query(default=20, ge=1, le=100),
):
    t0 = time.monotonic()

    # Verify the image exists
    if qdrant_service.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail="Image not found")

    hits = qdrant_service.search_similar(image_id, k=k)
    results = [_hit_to_search_result(h) for h in hits]
    elapsed_ms = (time.monotonic() - t0) * 1000

    return SearchResponse(
        results=results,
        total=len(results),
        query=f"similar:{image_id}",
        query_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

@router.get("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str = Query(..., min_length=1, description="Natural language query"),
    k: int = Query(default=20, ge=1, le=100),
    alpha: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense results (0 = sparse only, 1 = dense only)",
    ),
):
    t0 = time.monotonic()

    try:
        query_embedding = clip_service.encode_text(q)
        hits = qdrant_service.hybrid_search(
            query_embedding=query_embedding,
            query_text=q,
            k=k,
            alpha=alpha,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable. Is Qdrant running?")

    results = [_hit_to_hybrid_result(h) for h in hits]
    elapsed_ms = (time.monotonic() - t0) * 1000

    return HybridSearchResponse(
        results=results,
        total=len(results),
        query=q,
        query_time_ms=round(elapsed_ms, 2),
        alpha=alpha,
    )
