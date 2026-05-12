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
