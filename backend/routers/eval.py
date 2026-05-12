"""
Evaluation endpoint.

POST /eval/run  — run Recall@K and NDCG@K benchmarks against stored collection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.eval_runner import load_benchmark, recall_at_k, ndcg_at_k
from models.schemas import EvalMetrics
from services.clip_service import clip_service
from services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    benchmark_path: str = Field(..., description="Absolute or relative path to benchmark JSON")
    k_values: list[int] = Field(default=[5, 10])
    strategies: list[str] = Field(default=["dense", "hybrid"])


class QueryBreakdown(BaseModel):
    query: str
    strategy: str
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


class EvalResponse(BaseModel):
    metrics: list[EvalMetrics]
    per_query: list[QueryBreakdown]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/run", response_model=EvalResponse)
async def run_eval(body: EvalRequest):
    benchmark_path = Path(body.benchmark_path)
    if not benchmark_path.exists():
        raise HTTPException(status_code=400, detail=f"Benchmark file not found: {benchmark_path}")

    try:
        benchmark = load_benchmark(str(benchmark_path))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load benchmark: {exc}")

    if not benchmark:
        raise HTTPException(status_code=422, detail="Benchmark is empty")

    metrics_list: list[EvalMetrics] = []
    per_query_list: list[QueryBreakdown] = []

    for strategy in body.strategies:
        r5_scores: list[float] = []
        r10_scores: list[float] = []
        ndcg10_scores: list[float] = []

        for item in benchmark:
            query: str = item.get("query", "")
            relevant_ids: list[str] = item.get("relevant_ids", [])
            if not query or not relevant_ids:
                continue

            # Retrieve top-max(k_values) results
            max_k = max(body.k_values)
            query_embedding = clip_service.encode_text(query)

            if strategy == "hybrid":
                hits = qdrant_service.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    k=max_k,
                )
            else:
                hits = qdrant_service.search(query_embedding, k=max_k)

            retrieved_ids = [h["id"] for h in hits]

            breakdown_recall: dict[int, float] = {}
            breakdown_ndcg: dict[int, float] = {}
            for kv in body.k_values:
                rec = recall_at_k(retrieved_ids, relevant_ids, kv)
                ndcg = ndcg_at_k(retrieved_ids, relevant_ids, kv)
                breakdown_recall[kv] = rec
                breakdown_ndcg[kv] = ndcg

            per_query_list.append(
                QueryBreakdown(
                    query=query,
                    strategy=strategy,
                    recall_at_k=breakdown_recall,
                    ndcg_at_k=breakdown_ndcg,
                )
            )

            if 5 in body.k_values:
                r5_scores.append(breakdown_recall[5])
            if 10 in body.k_values:
                r10_scores.append(breakdown_recall[10])
                ndcg10_scores.append(breakdown_ndcg[10])

        num = len(r5_scores) or len(r10_scores) or 1
        metrics_list.append(
            EvalMetrics(
                recall_at_5=round(sum(r5_scores) / max(len(r5_scores), 1), 4),
                recall_at_10=round(sum(r10_scores) / max(len(r10_scores), 1), 4),
                ndcg_at_10=round(sum(ndcg10_scores) / max(len(ndcg10_scores), 1), 4),
                num_queries=len(benchmark),
                strategy=strategy,
            )
        )

    return EvalResponse(metrics=metrics_list, per_query=per_query_list)
