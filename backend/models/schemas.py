"""
Pydantic v2 schemas for VisualMind AI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tag / metadata models
# ---------------------------------------------------------------------------

class ImageTags(BaseModel):
    layout_type: str = Field(
        default="other",
        description="One of: dashboard, landing, onboarding, form, card, other",
    )
    color_mood: str = Field(
        default="minimal",
        description="One of: minimal, dark, colorful, warm, corporate, playful",
    )
    ui_patterns: list[str] = Field(
        default_factory=list,
        description="Detected UI patterns from the allowed vocabulary",
    )
    industry: str = Field(
        default="other",
        description="One of: fintech, saas, ecommerce, health, education, other",
    )
    complexity: str = Field(
        default="medium",
        description="One of: low, medium, high",
    )


# ---------------------------------------------------------------------------
# Core image record
# ---------------------------------------------------------------------------

class ImageRecord(BaseModel):
    id: str
    url: str
    thumbnail_url: str
    tags: ImageTags
    collection_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phash: str
    embedding_dim: int = 512


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    id: str
    score: float
    url: str
    thumbnail_url: str
    tags: ImageTags


class HybridSearchResult(SearchResult):
    dense_score: float
    sparse_score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str
    query_time_ms: float


class HybridSearchResponse(BaseModel):
    results: list[HybridSearchResult]
    total: int
    query: str
    query_time_ms: float
    alpha: float


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    image_ids: list[str]
    tags: dict[str, ImageTags]
    duplicates: list[str]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Browse (list all images)
# ---------------------------------------------------------------------------

class BrowseItem(BaseModel):
    id: str
    url: str
    thumbnail_url: str
    tags: ImageTags
    filename: Optional[str] = None
    created_at: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class BrowseResponse(BaseModel):
    images: list[BrowseItem]
    total: int


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalMetrics(BaseModel):
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    num_queries: int
    strategy: str


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    model: bool
    model_name: str
    image_count: int


class StatsResponse(BaseModel):
    total_images: int
    total_collections: int
    tag_distribution: dict
    index_size_mb: float
