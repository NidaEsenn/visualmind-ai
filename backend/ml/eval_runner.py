"""
Evaluation runner for VisualMind AI.

Can be imported as a module or run as a standalone script:

    python -m ml.eval_runner --benchmark_path eval/benchmark.json --strategy dense

Benchmark JSON format:
[
  {
    "query": "fintech dashboard with charts",
    "relevant_ids": ["img_abc12345", "img_def67890", ...]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> list[dict]:
    """Load benchmark JSON from *path* and return the list of query dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of query objects.")
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@K = |retrieved_ids[:k] ∩ relevant_ids| / |relevant_ids|

    Returns 0.0 if *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    NDCG@K using binary relevance.

    DCG  = Σ rel_i / log2(i + 2)   for i in 0..k-1
    IDCG = DCG of the ideal ranking (all relevant docs first)
    NDCG = DCG / IDCG
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(top_k)
        if doc_id in relevant_set
    )

    # Ideal DCG: min(|relevant|, k) hits at positions 0..k-1
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    benchmark: list[dict],
    strategy: str,
    k_values: list[int],
) -> dict[str, Any]:
    """
    Run the full evaluation loop for a given *strategy*.

    Imports services lazily so that this module can also be used in
    offline/analysis contexts where the ML stack is not available.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from services.clip_service import clip_service
    from services.qdrant_service import qdrant_service

    max_k = max(k_values)
    per_query_results: list[dict] = []

    aggregate: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_agg: dict[int, list[float]] = {k: [] for k in k_values}

    for item in benchmark:
        query: str = item.get("query", "")
        relevant_ids: list[str] = item.get("relevant_ids", [])
        if not query or not relevant_ids:
            continue

        query_embedding = clip_service.encode_text(query)

        if strategy == "hybrid":
            hits = qdrant_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                k=max_k,
            )
        else:
            hits = qdrant_service.search(query_embedding, k=max_k)

        retrieved = [h["id"] for h in hits]

        q_result: dict[str, Any] = {"query": query, "recall": {}, "ndcg": {}}
        for kv in k_values:
            rec = recall_at_k(retrieved, relevant_ids, kv)
            ndcg = ndcg_at_k(retrieved, relevant_ids, kv)
            q_result["recall"][kv] = round(rec, 4)
            q_result["ndcg"][kv] = round(ndcg, 4)
            aggregate[kv].append(rec)
            ndcg_agg[kv].append(ndcg)

        per_query_results.append(q_result)

    summary: dict[str, Any] = {
        "strategy": strategy,
        "num_queries": len(per_query_results),
        "per_query": per_query_results,
    }
    for kv in k_values:
        n = max(len(aggregate[kv]), 1)
        summary[f"recall_at_{kv}"] = round(sum(aggregate[kv]) / n, 4)
        summary[f"ndcg_at_{kv}"] = round(sum(ndcg_agg[kv]) / n, 4)

    return summary


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict], k_values: list[int]) -> None:
    header_parts = ["Strategy", "# Queries"]
    for kv in k_values:
        header_parts += [f"Recall@{kv}", f"NDCG@{kv}"]
    header = " | ".join(f"{h:<14}" for h in header_parts)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        row_parts = [r["strategy"], str(r["num_queries"])]
        for kv in k_values:
            row_parts.append(f"{r.get(f'recall_at_{kv}', 0):.4f}")
            row_parts.append(f"{r.get(f'ndcg_at_{kv}', 0):.4f}")
        print(" | ".join(f"{v:<14}" for v in row_parts))
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="VisualMind AI evaluation runner")
    parser.add_argument(
        "--benchmark_path",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--strategy",
        default="dense",
        choices=["dense", "hybrid", "both"],
        help="Search strategy to evaluate (default: dense)",
    )
    parser.add_argument(
        "--k_values",
        default="5,10",
        help="Comma-separated list of K values (default: 5,10)",
    )
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    strategies = ["dense", "hybrid"] if args.strategy == "both" else [args.strategy]

    benchmark = load_benchmark(args.benchmark_path)
    logger.info("Loaded %d queries from %s", len(benchmark), args.benchmark_path)

    all_results: list[dict] = []
    for strat in strategies:
        logger.info("Running evaluation with strategy: %s", strat)
        result = run_evaluation(benchmark, strat, k_values)
        all_results.append(result)

    print("\n=== Evaluation Results ===")
    print_comparison_table(all_results, k_values)

    # Save results
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    out_path = eval_dir / f"results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Results saved to %s", out_path)
