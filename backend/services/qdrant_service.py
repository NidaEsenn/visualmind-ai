"""
Qdrant vector-database service.

Manages the `visualmind_images` collection (cosine distance, 768-dim vectors).
Exposes upsert, dense search, similar-image search, and hybrid search via
reciprocal-rank fusion of dense CLIP embeddings + BM25-style keyword matching
over tag payload fields.

Singleton instance exposed as `qdrant_service`.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantService:
    """Thin wrapper around qdrant_client for VisualMind AI."""

    def __init__(self) -> None:
        url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection: str = os.getenv("QDRANT_COLLECTION", "visualmind_images")
        # Match dim to model: ViT-B/32=512, ViT-L/14=768. Read from env or auto-detect.
        _model = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
        _dim_map = {"ViT-B-32": 512, "ViT-B-16": 512, "ViT-L-14": 768, "ViT-H-14": 1024}
        self.dim: int = _dim_map.get(_model, 512)

        logger.info("Connecting to Qdrant at %s", url)
        self.client = QdrantClient(url=url, timeout=30)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            self.client.get_collection(self.collection)
            logger.info("Qdrant collection '%s' already exists.", self.collection)
        except (UnexpectedResponse, Exception):
            logger.info("Creating Qdrant collection '%s'.", self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_image(
        self, image_id: str, embedding: np.ndarray, payload: dict
    ) -> None:
        """Insert or update a single image point."""
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=self._id_to_uint(image_id),
                    vector=embedding.tolist(),
                    payload={**payload, "_str_id": image_id},
                )
            ],
        )

    def delete_image(self, image_id: str) -> bool:
        """Delete a point by image_id. Returns True if deleted."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(
                    points=[self._id_to_uint(image_id)]
                ),
            )
            return True
        except Exception as exc:
            logger.warning("Failed to delete image %s: %s", image_id, exc)
            return False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_image(self, image_id: str) -> dict | None:
        """Retrieve the payload for a single image by ID."""
        results = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._id_to_uint(image_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        point = results[0]
        payload = dict(point.payload or {})
        payload["id"] = image_id
        return payload

    def list_images(self, limit: int = 200, filters: dict | None = None) -> list[dict]:
        """Scroll all images (newest first), with optional tag filters."""
        qdrant_filter = self._build_filter(filters) if filters else None
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qdrant_filter,
        )
        items = []
        for point in points:
            payload = dict(point.payload or {})
            str_id = payload.get("_str_id", str(point.id))
            items.append({"id": str_id, "payload": payload})
        items.sort(key=lambda x: x["payload"].get("created_at", ""), reverse=True)
        return items

    def get_embedding(self, image_id: str) -> np.ndarray | None:
        """Retrieve the raw embedding vector for an image."""
        results = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._id_to_uint(image_id)],
            with_payload=False,
            with_vectors=True,
        )
        if not results:
            return None
        vec = results[0].vector
        if vec is None:
            return None
        return np.array(vec, dtype=np.float32)

    def get_collection_stats(self) -> dict:
        """Return basic stats: total count + rough index size estimate."""
        info = self.client.get_collection(self.collection)
        count = info.points_count or 0
        # Rough estimate: each point stores a 768-dim float32 vector (3 072 bytes)
        # plus ~512 bytes for payload overhead.
        index_size_mb = round((count * (self.dim * 4 + 512)) / (1024 ** 2), 2)
        return {
            "count": count,
            "index_size_mb": index_size_mb,
            "collection": self.collection,
        }

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Dense cosine-similarity search.

        Returns list of {id, score, payload}.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        return [self._hit_to_dict(h) for h in hits]

    def search_similar(self, image_id: str, k: int = 20) -> list[dict]:
        """Find images similar to image_id via its stored embedding."""
        embedding = self.get_embedding(image_id)
        if embedding is None:
            return []
        # Exclude the query image itself
        return self.search(embedding, k=k + 1)[1:]  # skip self

    # ------------------------------------------------------------------
    # Hybrid search (dense + BM25-style keyword, fused via RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 20,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Reciprocal-rank fusion of:
          - Dense CLIP cosine similarity (weight: alpha)
          - BM25-style keyword matching over concatenated tag fields (weight: 1-alpha)

        The sparse step uses keyword matching over stored payload text because
        Qdrant sparse-vector indexing requires a separate index setup that is out
        of scope here.
        """
        fetch_k = min(k * 5, 200)  # over-fetch for re-ranking

        # --- Dense results ---
        dense_hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=fetch_k,
            with_payload=True,
        )

        # --- Scroll all points for BM25 keyword matching ---
        # This approach works for small-to-medium collections.  For large
        # collections a proper sparse index should be added.
        sparse_scores: dict[str, float] = {}
        payloads_by_id: dict[str, dict] = {}

        scroll_result = self.client.scroll(
            collection_name=self.collection,
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        )
        points, _next = scroll_result
        keywords = query_text.lower().split()

        for point in points:
            str_id = (point.payload or {}).get("_str_id", str(point.id))
            payload = dict(point.payload or {})
            payloads_by_id[str_id] = payload
            tag_text = self._payload_to_tag_string(payload).lower()
            score = sum(tag_text.count(kw) for kw in keywords)
            sparse_scores[str_id] = score

        # --- RRF fusion ---
        RRF_K = 60  # standard RRF constant

        dense_rank: dict[str, int] = {}
        dense_score_map: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            str_id = (hit.payload or {}).get("_str_id", str(hit.id))
            dense_rank[str_id] = rank
            dense_score_map[str_id] = hit.score
            if str_id not in payloads_by_id:
                payloads_by_id[str_id] = dict(hit.payload or {})

        sorted_sparse = sorted(sparse_scores.items(), key=lambda x: -x[1])
        sparse_rank: dict[str, int] = {
            sid: rank for rank, (sid, _) in enumerate(sorted_sparse, start=1)
        }

        all_ids = set(dense_rank) | set(sparse_rank)
        fused: dict[str, float] = {}
        for sid in all_ids:
            dr = dense_rank.get(sid, fetch_k + 1)
            sr = sparse_rank.get(sid, fetch_k + 1)
            rrf = alpha / (RRF_K + dr) + (1 - alpha) / (RRF_K + sr)
            fused[sid] = rrf

        top_ids = sorted(fused, key=lambda x: -fused[x])[:k]

        results = []
        for sid in top_ids:
            payload = payloads_by_id.get(sid, {})
            results.append(
                {
                    "id": sid,
                    "score": fused[sid],
                    "dense_score": dense_score_map.get(sid, 0.0),
                    "sparse_score": sparse_scores.get(sid, 0.0),
                    "payload": payload,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _id_to_uint(image_id: str) -> int:
        """
        Convert an `img_XXXXXXXX` string ID to a stable unsigned integer
        suitable for Qdrant point IDs.
        """
        # Use the hex portion (8 chars) directly as an integer
        hex_part = image_id.replace("img_", "")
        try:
            return int(hex_part, 16)
        except ValueError:
            # Fallback: hash the whole string
            return abs(hash(image_id)) % (2 ** 63)

    @staticmethod
    def _hit_to_dict(hit: Any) -> dict:
        payload = dict(hit.payload or {})
        str_id = payload.get("_str_id", str(hit.id))
        return {"id": str_id, "score": hit.score, "payload": payload}

    @staticmethod
    def _payload_to_tag_string(payload: dict) -> str:
        """Concatenate all tag-related fields into a single searchable string."""
        parts: list[str] = []
        for key in ("layout_type", "color_mood", "industry", "complexity"):
            val = payload.get(key) or payload.get("tags", {}).get(key, "")
            if val:
                parts.append(str(val))
        patterns = payload.get("ui_patterns") or payload.get("tags", {}).get(
            "ui_patterns", []
        )
        if isinstance(patterns, list):
            parts.extend(patterns)
        return " ".join(parts)

    @staticmethod
    def _build_filter(filters: dict) -> qmodels.Filter:
        """Build a Qdrant filter. List values become OR conditions; cross-field is AND."""
        must = []
        for field, value in filters.items():
            if isinstance(value, list) and len(value) > 1:
                must.append(qmodels.Filter(should=[
                    qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=v))
                    for v in value
                ]))
            else:
                v = value[0] if isinstance(value, list) else value
                must.append(qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=v)))
        return qmodels.Filter(must=must)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

qdrant_service = QdrantService()
