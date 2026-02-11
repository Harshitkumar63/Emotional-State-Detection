"""
Train the Attention Fusion Model for Burnout Risk Prediction
==============================================================
This script trains the AttentionFusionNetwork using SYNTHETIC data.

How synthetic training data is generated:
-----------------------------------------
Since no publicly available dataset has aligned (text + audio + face)
samples labelled with burnout risk, we generate synthetic embeddings
that encode domain knowledge about the emotion → burnout mapping:

  - LOW RISK embeddings are drawn from distributions centered on
    "joy" and "neutral" emotion regions in embedding space.
  - MODERATE RISK embeddings cluster near "anger" and "surprise"
    regions (mixed signals, some concerning).
  - HIGH RISK embeddings cluster near "sadness" and "fear" regions
    (strong negative signals across modalities).

This is NOT equivalent to training on real data.  The model learns
the general pattern "negative emotions → higher risk" but lacks the
nuance of real human burnout data.

Limitation:
  The fusion model's accuracy is bounded by the quality of synthetic
  data.  For production deployment, fine-tune on real aligned datasets
  like MELD, CMU-MOSEI, or a custom burnout-annotated corpus.

  Despite this limitation, the architecture is sound — replace the
  synthetic data with real data and the model improves automatically.

Usage:
  python scripts/train_fusion.py
  python scripts/train_fusion.py --epochs 50 --lr 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion.attention_fusion import AttentionFusionNetwork
from src.utils.helpers import load_config, setup_logging

logger = setup_logging()


# ======================================================================
# Synthetic Data Generation
# ======================================================================

def generate_synthetic_embeddings(
    n_samples: int = 2000,
    embedding_dim: int = 768,
    seed: int = 42,
) -> tuple:
    """Generate synthetic embeddings with emotion-based structure.

    The key insight: real emotion model embeddings cluster in meaningful
    ways — joy embeddings are closer to each other than to sadness
    embeddings.  We simulate this structure using class-conditional
    Gaussian distributions with learned centers.

    Parameters
    ----------
    n_samples : int
        Total number of samples (split equally across 3 classes).
    embedding_dim : int
        Embedding dimension (768 to match real models).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple: (text_emb, voice_emb, face_emb, labels, modality_masks)
    """
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 3

    # Class-conditional centers (simulating emotion clustering)
    # These are random but FIXED — they create consistent structure
    centers = {
        0: rng.randn(embedding_dim) * 0.3 + 0.5,    # Low Risk: positive bias
        1: rng.randn(embedding_dim) * 0.3,            # Moderate: neutral
        2: rng.randn(embedding_dim) * 0.3 - 0.5,     # High Risk: negative bias
    }

    all_text = []
    all_voice = []
    all_face = []
    all_labels = []
    all_masks = []

    for label in range(3):
        center = centers[label]

        for _ in range(n_per_class):
            # Each modality gets a slightly different view of the same
            # "emotional state" — simulating how text, voice, and face
            # encode overlapping but distinct information.
            noise_scale = 0.3 + label * 0.1  # higher risk = noisier signals

            text_emb = center + rng.randn(embedding_dim) * noise_scale
            voice_emb = center + rng.randn(embedding_dim) * noise_scale * 1.2
            face_emb = center + rng.randn(embedding_dim) * noise_scale * 0.8

            # Randomly mask some modalities to train handling of missing data
            mask = [1, 1, 1]
            r = rng.random()
            if r < 0.15:
                # Text only
                mask = [1, 0, 0]
                voice_emb = np.zeros(embedding_dim)
                face_emb = np.zeros(embedding_dim)
            elif r < 0.25:
                # Voice only
                mask = [0, 1, 0]
                text_emb = np.zeros(embedding_dim)
                face_emb = np.zeros(embedding_dim)
            elif r < 0.35:
                # Face only
                mask = [0, 0, 1]
                text_emb = np.zeros(embedding_dim)
                voice_emb = np.zeros(embedding_dim)
            elif r < 0.45:
                # Text + Voice
                mask = [1, 1, 0]
                face_emb = np.zeros(embedding_dim)
            elif r < 0.55:
                # Text + Face
                mask = [1, 0, 1]
                voice_emb = np.zeros(embedding_dim)

            all_text.append(text_emb)
            all_voice.append(voice_emb)
            all_face.append(face_emb)
            all_labels.append(label)
            all_masks.append(mask)

    # Convert to tensors
    text_tensor = torch.FloatTensor(np.stack(all_text))
    voice_tensor = torch.FloatTensor(np.stack(all_voice))
    face_tensor = torch.FloatTensor(np.stack(all_face))
    label_tensor = torch.LongTensor(all_labels)
    mask_tensor = torch.FloatTensor(np.stack(all_masks))

    # Shuffle
    perm = torch.randperm(len(label_tensor))
    return (
        text_tensor[perm],
        voice_tensor[perm],
        face_tensor[perm],
        label_tensor[perm],
        mask_tensor[perm],
    )


# ======================================================================
# Training Loop
# ======================================================================

def train_fusion(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    n_samples: int = 3000,
    save_path: str = "checkpoints/fusion_model.pt",
):
    """Train the AttentionFusionNetwork on synthetic data.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for DataLoader.
    lr : float
        Learning rate for Adam optimizer.
    n_samples : int
        Number of synthetic samples to generate.
    save_path : str
        Where to save the trained model weights.
    """
    config = load_config()
    fusion_cfg = config.get("fusion", {})

    logger.info("=" * 60)
    logger.info("Training Attention Fusion Model")
    logger.info("=" * 60)

    # Generate data
    logger.info("Generating %d synthetic training samples...", n_samples)
    text_emb, voice_emb, face_emb, labels, masks = generate_synthetic_embeddings(
        n_samples=n_samples
    )

    # Train/val split (80/20)
    split_idx = int(len(labels) * 0.8)
    train_dataset = TensorDataset(
        text_emb[:split_idx], voice_emb[:split_idx], face_emb[:split_idx],
        labels[:split_idx], masks[:split_idx],
    )
    val_dataset = TensorDataset(
        text_emb[split_idx:], voice_emb[split_idx:], face_emb[split_idx:],
        labels[split_idx:], masks[split_idx:],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = AttentionFusionNetwork(
        embedding_dim=768,
        hidden_dim=fusion_cfg.get("hidden_dim", 256),
        num_classes=fusion_cfg.get("num_classes", 3),
        dropout=fusion_cfg.get("dropout", 0.3),
    )
    model.train()

    # Class weights to handle any imbalance
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_text, batch_voice, batch_face, batch_labels, batch_masks in train_loader:
            optimizer.zero_grad()

            # Build embeddings dict with masking
            embeddings = {}
            if batch_masks[:, 0].sum() > 0:
                embeddings["text"] = batch_text
            if batch_masks[:, 1].sum() > 0:
                embeddings["voice"] = batch_voice
            if batch_masks[:, 2].sum() > 0:
                embeddings["face"] = batch_face

            if not embeddings:
                continue

            output = model(embeddings, return_weights=False)
            loss = criterion(output["logits"], batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_labels)
            train_correct += (output["predicted_class"] == batch_labels).sum().item()
            train_total += len(batch_labels)

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_text, batch_voice, batch_face, batch_labels, batch_masks in val_loader:
                embeddings = {}
                if batch_masks[:, 0].sum() > 0:
                    embeddings["text"] = batch_text
                if batch_masks[:, 1].sum() > 0:
                    embeddings["voice"] = batch_voice
                if batch_masks[:, 2].sum() > 0:
                    embeddings["face"] = batch_face

                if not embeddings:
                    continue

                output = model(embeddings, return_weights=False)
                loss = criterion(output["logits"], batch_labels)

                val_loss += loss.item() * len(batch_labels)
                val_correct += (output["predicted_class"] == batch_labels).sum().item()
                val_total += len(batch_labels)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%%",
            epoch + 1, epochs, avg_train_loss, train_acc * 100,
            avg_val_loss, val_acc * 100,
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best model saved (val_acc=%.1f%%)", val_acc * 100)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Best validation accuracy: %.1f%%", best_val_acc * 100)
    logger.info("Model saved to: %s", save_path)
    logger.info("=" * 60)

    return model


# ======================================================================
# CLI Entry Point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Attention Fusion Model")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples", type=int, default=3000, help="Synthetic samples")
    parser.add_argument("--save", type=str, default="checkpoints/fusion_model.pt")

    args = parser.parse_args()

    train_fusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.samples,
        save_path=args.save,
    )
