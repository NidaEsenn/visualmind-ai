"""
Data collection script for VisualMind AI.

Scrapes design screenshots from Dribbble and Mobbin, then uses Claude vision
to generate text captions for CLIP fine-tuning.

Usage:
    python -m ml.scraper dribbble --output_dir data/raw --num_images 5000
    python -m ml.scraper mobbin   --output_dir data/raw --num_images 5000
    python -m ml.scraper captions --image_dir data/raw --output_file data/captions.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limiting helpers
# ---------------------------------------------------------------------------

def _rate_limited_get(
    client: httpx.Client,
    url: str,
    delay: float = 0.5,
    retries: int = 3,
    **kwargs: Any,
) -> httpx.Response | None:
    """GET *url* with exponential back-off on 429 / 5xx responses."""
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=30, **kwargs)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.warning("Rate limited on %s — sleeping %.1fs", url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(delay)
            return resp
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP error %s on %s (attempt %d)", exc.response.status_code, url, attempt + 1)
            time.sleep(delay * (2 ** attempt))
        except httpx.RequestError as exc:
            logger.warning("Request error on %s: %s (attempt %d)", url, exc, attempt + 1)
            time.sleep(delay * (2 ** attempt))
    logger.error("Failed to fetch %s after %d attempts.", url, retries)
    return None


def _save_image(content: bytes, dest: Path) -> bool:
    """Save raw image bytes to *dest*, converting to JPEG if needed. Returns True on success."""
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.save(str(dest), format="JPEG", quality=90)
        return True
    except Exception as exc:
        logger.warning("Could not save image to %s: %s", dest, exc)
        return False


# ---------------------------------------------------------------------------
# Dribbble scraper
# ---------------------------------------------------------------------------

def scrape_dribbble(
    output_dir: str,
    num_images: int = 5000,
    api_token: str | None = None,
) -> None:
    """
    Fetch UI shot images from Dribbble.

    If *api_token* is not supplied it is read from the DRIBBBLE_API_TOKEN
    environment variable.  Without a token the function logs a warning and
    returns without scraping.

    Dribbble API reference: https://developer.dribbble.com/v2/shots/
    """
    token = api_token or os.getenv("DRIBBBLE_API_TOKEN", "")
    if not token:
        logger.warning(
            "DRIBBBLE_API_TOKEN not set — skipping Dribbble scraping. "
            "Obtain a token at https://dribbble.com/oauth/applications/new"
        )
        return

    out_path = Path(output_dir) / "dribbble"
    out_path.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {token}"}
    collected = 0
    page = 1
    per_page = 100  # Dribbble max

    logger.info("Starting Dribbble scrape — target: %d images", num_images)

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        while collected < num_images:
            resp = _rate_limited_get(
                client,
                "https://api.dribbble.com/v2/shots",
                params={"page": page, "per_page": per_page, "sort": "popular"},
                delay=1.0,
            )
            if resp is None:
                break

            shots: list[dict] = resp.json()
            if not shots:
                logger.info("No more Dribbble shots on page %d.", page)
                break

            for shot in shots:
                if collected >= num_images:
                    break
                try:
                    img_url = shot["images"].get("hidpi") or shot["images"].get("normal")
                    if not img_url:
                        continue
                    shot_id = str(shot["id"])
                    dest = out_path / f"dribbble_{shot_id}.jpg"
                    if dest.exists():
                        collected += 1
                        continue

                    img_resp = _rate_limited_get(client, img_url, delay=0.3)
                    if img_resp and _save_image(img_resp.content, dest):
                        collected += 1
                        if collected % 100 == 0:
                            logger.info("Dribbble: collected %d / %d", collected, num_images)
                except Exception as exc:
                    logger.warning("Failed to process Dribbble shot: %s", exc)

            page += 1

    logger.info("Dribbble scrape complete: %d images saved to %s", collected, out_path)


# ---------------------------------------------------------------------------
# Mobbin scraper
# ---------------------------------------------------------------------------

def scrape_mobbin(
    output_dir: str,
    num_images: int = 5000,
) -> None:
    """
    Fetch UI screenshots from Mobbin.

    Mobbin does not have a public API.  This stub demonstrates the correct
    structure for rate-limited HTTP scraping; replace the placeholder URL
    and parsing logic with the actual Mobbin API/CDN details once you have
    access credentials (https://mobbin.com/).
    """
    out_path = Path(output_dir) / "mobbin"
    out_path.mkdir(parents=True, exist_ok=True)

    mobbin_api_key = os.getenv("MOBBIN_API_KEY", "")
    if not mobbin_api_key:
        logger.warning(
            "MOBBIN_API_KEY not set — skipping Mobbin scraping. "
            "Obtain credentials at https://mobbin.com/"
        )
        return

    logger.info("Starting Mobbin scrape — target: %d images", num_images)
    collected = 0
    page = 1

    with httpx.Client(
        headers={"Authorization": f"Bearer {mobbin_api_key}"},
        follow_redirects=True,
    ) as client:
        while collected < num_images:
            # Replace with actual Mobbin API endpoint
            resp = _rate_limited_get(
                client,
                "https://api.mobbin.com/v1/screenshots",
                params={"page": page, "limit": 50, "category": "all"},
                delay=1.5,
            )
            if resp is None:
                break

            data: dict = resp.json()
            items: list[dict] = data.get("data", [])
            if not items:
                break

            for item in items:
                if collected >= num_images:
                    break
                try:
                    img_url = item.get("image_url", "")
                    item_id = str(item.get("id", f"unknown_{collected}"))
                    dest = out_path / f"mobbin_{item_id}.jpg"
                    if dest.exists():
                        collected += 1
                        continue
                    img_resp = _rate_limited_get(client, img_url, delay=0.3)
                    if img_resp and _save_image(img_resp.content, dest):
                        collected += 1
                        if collected % 100 == 0:
                            logger.info("Mobbin: collected %d / %d", collected, num_images)
                except Exception as exc:
                    logger.warning("Failed to process Mobbin item: %s", exc)

            page += 1

    logger.info("Mobbin scrape complete: %d images saved to %s", collected, out_path)


# ---------------------------------------------------------------------------
# Caption generation with Claude
# ---------------------------------------------------------------------------

_CAPTION_SYSTEM_PROMPT = """\
You are a design analyst. Look at this UI screenshot and write a concise, descriptive caption
(1–2 sentences) suitable for training a CLIP model.

Focus on:
- The overall layout and purpose (e.g., "A dark-themed SaaS analytics dashboard...")
- Key UI elements visible (sidebar, charts, cards, etc.)
- Color palette and visual style
- Industry context if apparent

Return ONLY the caption text, no other commentary.\
"""


def build_caption_pairs(
    image_dir: str,
    output_file: str,
    batch_size: int = 10,
) -> None:
    """
    Use Claude vision to generate text captions for all images in *image_dir*.

    Saves results as a JSON list of {"image_path": ..., "caption": ...} dicts
    to *output_file*.  Already-captioned images are skipped on re-runs.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set — cannot generate captions.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    image_path = Path(image_dir)
    out_path = Path(output_file)

    # Load existing captions to support resuming
    existing: dict[str, str] = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as fh:
            for item in json.load(fh):
                existing[item["image_path"]] = item["caption"]

    image_files = sorted(
        p for p in image_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    logger.info(
        "Caption generation: %d images found (%d already captioned).",
        len(image_files), len(existing),
    )

    results: list[dict] = list({"image_path": k, "caption": v} for k, v in existing.items())
    new_count = 0

    for i, img_file in enumerate(image_files):
        str_path = str(img_file)
        if str_path in existing:
            continue

        # Load and base64-encode
        try:
            img = Image.open(str_path).convert("RGB")
            # Resize for efficiency
            if img.width > 1024:
                ratio = 1024 / img.width
                img = img.resize((1024, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("Could not process %s: %s", str_path, exc)
            continue

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=256,
                    system=_CAPTION_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": "Describe this UI screenshot."},
                            ],
                        }
                    ],
                )
                caption = response.content[0].text.strip()
                results.append({"image_path": str_path, "caption": caption})
                new_count += 1
                time.sleep(0.4)  # gentle rate limiting
                break

            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
            except Exception as exc:
                logger.warning("Caption error for %s (attempt %d): %s", str_path, attempt + 1, exc)
                if attempt == 2:
                    results.append({"image_path": str_path, "caption": ""})
                time.sleep(1)

        # Persist every batch_size images
        if new_count > 0 and new_count % batch_size == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            logger.info("Progress: %d / %d captioned — saved.", new_count, len(image_files) - len(existing))

    # Final save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Caption generation complete: %d new captions written to %s", new_count, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="VisualMind AI data scraper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dribbble
    p_drib = subparsers.add_parser("dribbble", help="Scrape Dribbble shots")
    p_drib.add_argument("--output_dir", default="data/raw")
    p_drib.add_argument("--num_images", type=int, default=5000)

    # mobbin
    p_mob = subparsers.add_parser("mobbin", help="Scrape Mobbin screenshots")
    p_mob.add_argument("--output_dir", default="data/raw")
    p_mob.add_argument("--num_images", type=int, default=5000)

    # captions
    p_cap = subparsers.add_parser("captions", help="Generate CLIP captions with Claude")
    p_cap.add_argument("--image_dir", required=True)
    p_cap.add_argument("--output_file", default="data/captions.json")
    p_cap.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    if args.command == "dribbble":
        scrape_dribbble(args.output_dir, args.num_images)
    elif args.command == "mobbin":
        scrape_mobbin(args.output_dir, args.num_images)
    elif args.command == "captions":
        build_caption_pairs(args.image_dir, args.output_file, args.batch_size)
