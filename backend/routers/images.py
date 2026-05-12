"""
Image upload, retrieval, and deletion endpoints.

GET  /images          — list all images (browseable gallery)
POST /images/upload   — ingest one or more image files
GET  /images/{id}     — fetch a single ImageRecord
DELETE /images/{id}   — remove from Qdrant + filesystem
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from models.schemas import BrowseItem, BrowseResponse, ImageRecord, ImageTags, UploadResponse
from services.clip_service import clip_service
from services.qdrant_service import qdrant_service
from services.tagging_service import tagging_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
THUMB_MAX_WIDTH = 400


def _storage_root() -> Path:
    path = os.getenv("IMAGE_STORAGE_PATH", "./data/images")
    return Path(path)


def _image_id(content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"img_{digest[:8]}"


def _phash_str(image: Image.Image) -> str:
    return str(imagehash.phash(image))


def _hamming(a: str, b: str) -> int:
    ha = imagehash.hex_to_hash(a)
    hb = imagehash.hex_to_hash(b)
    return ha - hb


def _save_image(image: Image.Image, dest: Path, ext: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if ext.lower() in ("jpg", "jpeg") else ext.upper()
    image.save(str(dest), format=fmt)


def _save_thumbnail(image: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = image.copy()
    if thumb.width > THUMB_MAX_WIDTH:
        ratio = THUMB_MAX_WIDTH / thumb.width
        thumb = thumb.resize(
            (THUMB_MAX_WIDTH, int(thumb.height * ratio)), Image.LANCZOS
        )
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    thumb.save(str(dest), format="JPEG", quality=85)


def _existing_phashes() -> list[str]:
    """Scroll through Qdrant and collect all stored pHashes."""
    try:
        points, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return [
            p.payload.get("phash", "")
            for p in points
            if p.payload and p.payload.get("phash")
        ]
    except Exception as exc:
        logger.warning("Could not fetch existing pHashes: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Browse / list all
# ---------------------------------------------------------------------------

def _item_to_browse(item: dict) -> BrowseItem:
    payload = item.get("payload", {})
    tags_data = payload.get("tags") or {
        "layout_type": payload.get("layout_type", "other"),
        "color_mood": payload.get("color_mood", "minimal"),
        "ui_patterns": payload.get("ui_patterns", []),
        "industry": payload.get("industry", "other"),
        "complexity": payload.get("complexity", "medium"),
    }
    return BrowseItem(
        id=item["id"],
        url=payload.get("url", ""),
        thumbnail_url=payload.get("thumbnail_url", ""),
        tags=ImageTags(**tags_data),
        filename=payload.get("filename"),
        created_at=payload.get("created_at"),
        width=payload.get("width"),
        height=payload.get("height"),
    )


@router.get("", response_model=BrowseResponse)
async def list_images(
    limit: int = Query(default=200, ge=1, le=500),
    filters: Optional[str] = Query(
        default=None,
        description='JSON filter object, e.g. {"layout_type":["dashboard"]}',
    ),
):
    filter_dict: Optional[dict] = None
    if filters:
        try:
            filter_dict = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="filters must be valid JSON")

    try:
        items = qdrant_service.list_images(limit=limit, filters=filter_dict)
    except Exception as exc:
        logger.error("Browse failed: %s", exc)
        raise HTTPException(status_code=503, detail="Storage service unavailable.")

    images = [_item_to_browse(item) for item in items]
    return BrowseResponse(images=images, total=len(images))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_images(
    files: list[UploadFile] = File(...),
    collection_id: Optional[str] = Query(default=None),
):
    t0 = time.monotonic()
    storage = _storage_root()
    thumbs_dir = storage / "thumbs"
    storage.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    existing_phashes = _existing_phashes()

    image_ids: list[str] = []
    tags_map: dict[str, ImageTags] = {}
    duplicates: list[str] = []

    for upload in files:
        content = await upload.read()

        # --- Validate format ---
        try:
            pil = Image.open(__import__("io").BytesIO(content))
            pil.verify()  # check header integrity
            pil = Image.open(__import__("io").BytesIO(content))  # reopen after verify
        except Exception as exc:
            logger.warning("Invalid image file %s: %s", upload.filename, exc)
            continue

        if pil.format not in ALLOWED_FORMATS:
            logger.warning(
                "Rejected %s: unsupported format %s", upload.filename, pil.format
            )
            continue

        # --- pHash deduplication ---
        ph = _phash_str(pil)
        is_dup = any(_hamming(ph, existing) < 10 for existing in existing_phashes)
        if is_dup:
            logger.info("Duplicate detected for %s — skipping.", upload.filename)
            duplicates.append(upload.filename or "unknown")
            continue

        # --- Compute image ID ---
        img_id = _image_id(content)

        # --- Determine file extension ---
        fmt = pil.format or "JPEG"
        ext_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}
        ext = ext_map.get(fmt, "jpg")

        orig_path = storage / f"{img_id}.{ext}"
        thumb_path = thumbs_dir / f"{img_id}.jpg"

        # --- Save files ---
        _save_image(pil, orig_path, ext)
        _save_thumbnail(pil, thumb_path)

        # --- CLIP encoding ---
        if pil.mode != "RGB":
            clip_input = pil.convert("RGB")
        else:
            clip_input = pil
        embedding: np.ndarray = clip_service.encode_image(clip_input)

        # --- VLM tagging ---
        image_tags: ImageTags = tagging_service.tag_image(clip_input)

        # --- Build Qdrant payload ---
        payload = {
            "url": f"/static/images/{img_id}.{ext}",
            "thumbnail_url": f"/static/images/thumbs/{img_id}.jpg",
            "phash": ph,
            "collection_id": collection_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_dim": clip_service.embedding_dim,
            # Flat tags for BM25 matching
            "layout_type": image_tags.layout_type,
            "color_mood": image_tags.color_mood,
            "ui_patterns": image_tags.ui_patterns,
            "industry": image_tags.industry,
            "complexity": image_tags.complexity,
            # Nested tags for structured access
            "tags": image_tags.model_dump(),
        }

        qdrant_service.upsert_image(img_id, embedding, payload)

        existing_phashes.append(ph)
        image_ids.append(img_id)
        tags_map[img_id] = image_tags

    elapsed_ms = (time.monotonic() - t0) * 1000
    return UploadResponse(
        image_ids=image_ids,
        tags=tags_map,
        duplicates=duplicates,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

@router.get("/{image_id}", response_model=ImageRecord)
async def get_image(image_id: str):
    payload = qdrant_service.get_image(image_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Image not found")

    tags_data = payload.get("tags") or {
        "layout_type": payload.get("layout_type", "other"),
        "color_mood": payload.get("color_mood", "minimal"),
        "ui_patterns": payload.get("ui_patterns", []),
        "industry": payload.get("industry", "other"),
        "complexity": payload.get("complexity", "medium"),
    }

    created_raw = payload.get("created_at")
    created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.now(timezone.utc)

    return ImageRecord(
        id=image_id,
        url=payload.get("url", ""),
        thumbnail_url=payload.get("thumbnail_url", ""),
        tags=ImageTags(**tags_data),
        collection_id=payload.get("collection_id"),
        created_at=created_at,
        phash=payload.get("phash", ""),
        embedding_dim=payload.get("embedding_dim", 768),
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{image_id}")
async def delete_image(image_id: str):
    payload = qdrant_service.get_image(image_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove files from disk
    storage = _storage_root()
    for candidate in storage.glob(f"{image_id}.*"):
        candidate.unlink(missing_ok=True)
    thumb = storage / "thumbs" / f"{image_id}.jpg"
    thumb.unlink(missing_ok=True)

    deleted = qdrant_service.delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete from Qdrant")

    return {"deleted": True, "id": image_id}
