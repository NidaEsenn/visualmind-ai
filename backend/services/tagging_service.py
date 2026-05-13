"""
VLM auto-tagging service using Google Gemini Flash (free tier).

Converts a PIL image to bytes, sends it to Gemini 2.0 Flash with a structured
prompt, and parses the JSON response into an ImageTags object.

Singleton instance exposed as `tagging_service`.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from models.schemas import ImageTags

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a design analyst. Analyze this UI screenshot and extract structured metadata.

Return ONLY valid JSON with this exact structure:
{
  "layout_type": "<dashboard|landing|onboarding|form|card|other>",
  "color_mood": "<minimal|dark|colorful|warm|corporate|playful>",
  "ui_patterns": ["<pattern1>", "<pattern2>"],
  "industry": "<fintech|saas|ecommerce|health|education|social|travel|media|productivity|crypto|other>",
  "complexity": "<low|medium|high>"
}

Industry definitions — pick the closest match, never default to "other" if a specific category fits:
- health: medical, healthcare, telemedicine, hospital, patient, doctor, clinic, pharmacy, wellness, fitness
- fintech: banking, payments, finance, investment, insurance, accounting, crypto exchange, wallet
- saas: software tools, B2B platforms, developer tools, API dashboards, admin panels, CRM, analytics
- ecommerce: online store, shopping, product listings, cart, marketplace, retail
- education: learning platform, courses, LMS, tutoring, school, university, e-learning
- social: social network, messaging, community, dating, chat, feed, profiles
- travel: flights, hotels, booking, maps, tourism, transportation, itinerary
- media: streaming, news, podcast, video, music, publishing, content
- productivity: task management, notes, calendar, project management, collaboration, time tracking
- crypto: blockchain, NFT, DeFi, token, Web3, wallet, exchange
- other: only if no above category fits at all

For ui_patterns, choose from: cards, sidebar, hero, modal, nav, table, chart,
progress-bar, illustration, avatar, form-fields, breadcrumb, tabs, carousel,
pricing-table, testimonial, cta, footer, search-bar, notification, empty-state

Return only the JSON object, no other text."""

_DEFAULT_TAGS = ImageTags(
    layout_type="other",
    color_mood="minimal",
    ui_patterns=[],
    industry="other",
    complexity="medium",
)

ALLOWED_PATTERNS = {
    "cards", "sidebar", "hero", "modal", "nav", "table", "chart",
    "progress-bar", "illustration", "avatar", "form-fields", "breadcrumb",
    "tabs", "carousel", "pricing-table", "testimonial", "cta", "footer",
    "search-bar", "notification", "empty-state",
}

MAX_RETRIES = 2


class TaggingService:
    """Calls Google Gemini Flash to extract structured design tags from an image."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set — tagging will return defaults.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
        logger.info("Tagging service ready (gemini-2.5-flash)")

    def tag_image(self, image: Image.Image) -> ImageTags:
        """Analyse a PIL image with Gemini vision and return structured ImageTags."""
        if self.client is None:
            return ImageTags(**_DEFAULT_TAGS.model_dump())

        img_bytes = self._image_to_bytes(image)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        SYSTEM_PROMPT,
                    ],
                )
                raw_text = response.text.strip()
                logger.debug("Gemini raw response: %s", raw_text)
                return self._parse_tags(raw_text)

            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "quota" in err_str.lower():
                    logger.warning("Rate limit on attempt %d: %s", attempt + 1, exc)
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                logger.error("Gemini API error: %s — using defaults.", exc)
                return ImageTags(**_DEFAULT_TAGS.model_dump())

        return ImageTags(**_DEFAULT_TAGS.model_dump())

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        """Resize large images and convert to JPEG bytes."""
        if image.width > 1024:
            ratio = 1024 / image.width
            image = image.resize((1024, int(image.height * ratio)), Image.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def _parse_tags(raw: str) -> ImageTags:
        """Parse JSON output into an ImageTags object."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse Gemini JSON: %s | raw=%r", exc, raw)
            return ImageTags(**_DEFAULT_TAGS.model_dump())

        layout_type = data.get("layout_type", "other")
        if layout_type not in {"dashboard", "landing", "onboarding", "form", "card", "other"}:
            layout_type = "other"

        color_mood = data.get("color_mood", "minimal")
        if color_mood not in {"minimal", "dark", "colorful", "warm", "corporate", "playful"}:
            color_mood = "minimal"

        industry = data.get("industry", "other")
        if industry not in {
            "fintech", "saas", "ecommerce", "health", "education",
            "social", "travel", "media", "productivity", "crypto", "other"
        }:
            industry = "other"

        complexity = data.get("complexity", "medium")
        if complexity not in {"low", "medium", "high"}:
            complexity = "medium"

        raw_patterns = data.get("ui_patterns", [])
        if not isinstance(raw_patterns, list):
            raw_patterns = []
        ui_patterns = [p for p in raw_patterns if p in ALLOWED_PATTERNS]

        return ImageTags(
            layout_type=layout_type,
            color_mood=color_mood,
            ui_patterns=ui_patterns,
            industry=industry,
            complexity=complexity,
        )


tagging_service = TaggingService()
