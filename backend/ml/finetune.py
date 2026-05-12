"""
LoRA fine-tuning script for the CLIP ViT-L/14 model on design UI images.

Usage:
    python -m ml.finetune --image_dir data/images --captions_file data/captions.json

The script requires a captions JSON file in the format:
[
  {"image_path": "data/images/img_abc12345.jpg", "caption": "..."},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fine-tuning configuration
# ---------------------------------------------------------------------------

FINETUNE_CONFIG: dict[str, Any] = {
    # Model
    "model_name": "ViT-L-14",
    "pretrained": "openai",
    # LoRA
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Training
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "max_grad_norm": 1.0,
    # Data
    "val_split": 0.1,
    "image_size": 224,
    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_every_n_epochs": 2,
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_metric": "val_recall_5",
    # Hardware
    "device": "auto",  # auto → mps > cuda > cpu
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UIImageDataset(Dataset):
    """Paired (image, caption) dataset for CLIP contrastive fine-tuning."""

    def __init__(
        self,
        pairs: list[dict],
        preprocess,
        tokenizer,
    ) -> None:
        self.pairs = pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        image_path = pair["image_path"]
        caption = pair["caption"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", image_path, exc)
            # Return a blank image on failure
            image_tensor = torch.zeros(3, FINETUNE_CONFIG["image_size"], FINETUNE_CONFIG["image_size"])

        text_tokens = self.tokenizer([caption])[0]
        return image_tensor, text_tokens


def build_dataset(image_dir: str, captions_file: str) -> Dataset:
    """
    Build a UIImageDataset from a directory of images and a captions JSON file.

    The captions file should contain a list of {"image_path": ..., "caption": ...} dicts.
    If image_path entries are relative, they are resolved against image_dir.
    """
    captions_path = Path(captions_file)
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    with open(captions_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    pairs: list[dict] = []
    for item in raw:
        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = Path(image_dir) / img_path
        if img_path.exists():
            pairs.append({"image_path": str(img_path), "caption": item["caption"]})
        else:
            logger.warning("Image not found, skipping: %s", img_path)

    logger.info("Dataset: %d valid pairs", len(pairs))

    _, _, preprocess = open_clip.create_model_and_transforms(
        FINETUNE_CONFIG["model_name"], pretrained=FINETUNE_CONFIG["pretrained"]
    )
    tokenizer = open_clip.get_tokenizer(FINETUNE_CONFIG["model_name"])
    return UIImageDataset(pairs, preprocess, tokenizer)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for an existing nn.Linear layer."""

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Replace linear projection layers inside transformer attention blocks
    with LoRA-wrapped equivalents.  The base model weights are frozen;
    only LoRA parameters are trainable.
    """
    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in model.named_modules():
        # Target attention projection layers (q, k, v, out_proj)
        if isinstance(module, nn.MultiheadAttention):
            for proj_name in ("in_proj_weight",):
                # open_clip uses in_proj_weight for combined QKV
                pass  # handled via sub-modules below

        # Target Linear layers inside ResidualAttentionBlock / attention layers
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(
                kw in child_name for kw in ("q_proj", "k_proj", "v_proj", "out_proj", "c_proj", "c_fc")
            ):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %d layers replaced | trainable params: %d / %d (%.2f%%)",
        replaced, trainable, total, 100 * trainable / max(total, 1),
    )
    return model


# ---------------------------------------------------------------------------
# Contrastive loss (CLIP-style InfoNCE)
# ---------------------------------------------------------------------------

def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# ---------------------------------------------------------------------------
# Recall@K helper (validation)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_recall_at_k(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k: int = 5,
) -> float:
    all_image_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []

    model.eval()
    for images, texts in dataloader:
        images = images.to(device)
        texts = texts.to(device)
        img_f = model.encode_image(images)
        txt_f = model.encode_text(texts)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        all_image_feats.append(img_f)
        all_text_feats.append(txt_f)

    img_matrix = torch.cat(all_image_feats)
    txt_matrix = torch.cat(all_text_feats)
    sims = img_matrix @ txt_matrix.T

    n = sims.shape[0]
    top_k_indices = sims.topk(k, dim=1).indices
    hits = sum(1 for i in range(n) if i in top_k_indices[i].tolist())
    return hits / n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict | None = None) -> None:
    """
    Main fine-tuning loop.

    Args:
        config: Override dict merged on top of FINETUNE_CONFIG.
                Must include at minimum `image_dir` and `captions_file`.
    """
    cfg = {**FINETUNE_CONFIG, **(config or {})}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # --- Device ---
    if cfg["device"] == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg["device"])
    logger.info("Training on %s", device)

    # --- Model ---
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["pretrained"]
    )
    model = apply_lora(model, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"])
    model = model.to(device)

    # --- Dataset ---
    full_dataset = build_dataset(cfg["image_dir"], cfg["captions_file"])
    n_val = max(1, int(len(full_dataset) * cfg["val_split"]))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Checkpoint dir ---
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training ---
    best_val_metric = -1.0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            texts = texts.to(device)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(texts)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            loss = contrastive_loss(img_feats, txt_feats, model.logit_scale.exp())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                logger.info(
                    "Epoch %d | Step %d | Loss %.4f | LR %.2e",
                    epoch, global_step, loss.item(), scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # Validation Recall@5
        val_recall = compute_recall_at_k(model, val_loader, device, k=5)
        logger.info(
            "Epoch %d complete | Avg loss: %.4f | Val Recall@5: %.4f | Time: %.1fs",
            epoch, avg_loss, val_recall, elapsed,
        )

        # Periodic checkpoint
        if epoch % cfg["save_every_n_epochs"] == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_recall_5": val_recall,
                    "config": cfg,
                },
                str(ckpt_path),
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

        # Early stopping
        if val_recall > best_val_metric:
            best_val_metric = val_recall
            epochs_without_improvement = 0
            # Save best model
            best_path = ckpt_dir / "best_model.pt"
            torch.save({"model_state_dict": model.state_dict(), "val_recall_5": val_recall}, str(best_path))
            logger.info("New best model (Recall@5=%.4f) saved to %s", val_recall, best_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d epoch(s) (patience=%d)",
                epochs_without_improvement, cfg["early_stopping_patience"],
            )
            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete. Best Val Recall@5: %.4f", best_val_metric)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for VisualMind AI")
    parser.add_argument("--image_dir", required=True, help="Directory containing training images")
    parser.add_argument("--captions_file", required=True, help="Path to captions JSON file")
    parser.add_argument("--batch_size", type=int, default=FINETUNE_CONFIG["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=FINETUNE_CONFIG["num_epochs"])
    parser.add_argument("--lr", type=float, default=FINETUNE_CONFIG["learning_rate"])
    parser.add_argument("--lora_rank", type=int, default=FINETUNE_CONFIG["lora_rank"])
    parser.add_argument("--checkpoint_dir", default=FINETUNE_CONFIG["checkpoint_dir"])
    args = parser.parse_args()

    overrides = {
        "image_dir": args.image_dir,
        "captions_file": args.captions_file,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "checkpoint_dir": args.checkpoint_dir,
    }
    train(config=overrides)
