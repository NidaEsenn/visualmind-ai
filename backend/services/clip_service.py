"""
CLIP embedding service using open_clip (ViT-L/14, 768-dim).

Singleton instance exposed as `clip_service`.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

logger = logging.getLogger(__name__)


# Descriptive prompts used to build per-industry reference embeddings.
# These are passed through encode_text's UI template, so they read as:
# "a screenshot of a <description> user interface design"
_INDUSTRY_DESCRIPTIONS: dict[str, str] = {
    "health":       "healthcare medical doctor patient hospital clinic telemedicine wellness",
    "fintech":      "banking financial payments investment trading portfolio insurance",
    "saas":         "software B2B dashboard analytics CRM data management admin panel",
    "ecommerce":    "online shopping store product listing cart checkout marketplace retail",
    "education":    "learning course lesson student teacher LMS e-learning curriculum",
    "social":       "social network profile feed messaging community followers chat",
    "travel":       "travel booking flight hotel destination tourism map itinerary",
    "media":        "streaming video music news content publishing podcast entertainment",
    "productivity": "task project management notes calendar collaboration workspace",
    "crypto":       "cryptocurrency blockchain NFT DeFi token wallet exchange Web3",
    "other":        "generic web application",
}


class CLIPService:
    """Wraps open_clip ViT-L/14 for image and text encoding."""

    def __init__(self) -> None:
        model_name: str = os.getenv("CLIP_MODEL_NAME", "ViT-L-14")
        pretrained: str = os.getenv("CLIP_PRETRAINED", "openai")
        checkpoint: str = os.getenv("CLIP_CHECKPOINT", "")

        # Device priority: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info("Loading CLIP model %s (%s) on %s", model_name, pretrained, self.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load fine-tuned checkpoint if provided
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if ckpt_path.exists():
                logger.info("Loading fine-tuned checkpoint from %s", ckpt_path)
                state = torch.load(str(ckpt_path), map_location=self.device)
                # Support both raw state_dict and checkpoint dicts
                state_dict = state.get("model_state_dict", state)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning("CLIP_CHECKPOINT set but file not found: %s", checkpoint)

        self.model_name = model_name
        # Dimension depends on model: ViT-B/32 → 512, ViT-L/14 → 768
        _dim_map = {"ViT-B-32": 512, "ViT-B-16": 512, "ViT-L-14": 768, "ViT-H-14": 1024}
        self.embedding_dim = _dim_map.get(model_name, 512)

        # Lazily populated by classify_industry on first call
        self._industry_embeddings: dict[str, np.ndarray] | None = None

        logger.info("CLIP service ready — embedding dim: %d", self.embedding_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalised 768-dim vector for a single PIL image."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().float().numpy()

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        """Return an L2-normalised vector for a text query.

        Wrapping the query in a UI-context template significantly improves
        CLIP's text-image alignment for design search queries.
        """
        prompt = f"a screenshot of a {query} user interface design"
        tokens = self.tokenizer([prompt]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().float().numpy()

    def classify_industry(self, image_embedding: np.ndarray, min_gap: float = 0.02) -> str:
        """Return the closest industry label for a pre-computed image embedding.

        Compares the image vector against reference text embeddings for each
        industry. Returns "other" if the best match does not beat the "other"
        baseline by at least min_gap — avoids low-confidence overrides.
        """
        if self._industry_embeddings is None:
            logger.info("Building industry reference embeddings…")
            self._industry_embeddings = {
                industry: self.encode_text(desc)
                for industry, desc in _INDUSTRY_DESCRIPTIONS.items()
            }

        scores = {
            ind: float(np.dot(image_embedding, emb))
            for ind, emb in self._industry_embeddings.items()
        }
        other_score = scores["other"]
        best = max((ind for ind in scores if ind != "other"), key=lambda k: scores[k])

        if scores[best] - other_score >= min_gap:
            logger.debug(
                "CLIP industry: %s (%.3f) vs other (%.3f)",
                best, scores[best], other_score,
            )
            return best
        return "other"

    @torch.inference_mode()
    def batch_encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return shape (N, 768) array of L2-normalised image embeddings."""
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

clip_service = CLIPService()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PIL import Image as PILImage

    logging.basicConfig(level=logging.INFO)

    # Create a tiny test image
    test_img = PILImage.new("RGB", (224, 224), color=(128, 64, 200))
    query = "a minimal dashboard UI with charts"

    img_vec = clip_service.encode_image(test_img)
    txt_vec = clip_service.encode_text(query)

    cosine_sim = float(np.dot(img_vec, txt_vec))
    print(f"Image embedding shape : {img_vec.shape}")
    print(f"Text  embedding shape : {txt_vec.shape}")
    print(f"Cosine similarity     : {cosine_sim:.4f}")

    sys.exit(0)
