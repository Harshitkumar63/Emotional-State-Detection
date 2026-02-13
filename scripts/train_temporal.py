"""
Train the Temporal Burnout Predictor (GRU)
===========================================
Trains a GRU-based model to predict burnout risk TRENDS from sequences
of past assessment results.

Usage:
  python scripts/train_temporal.py
  python scripts/train_temporal.py --epochs 40 --sequences 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import (
    TemporalBurnoutPredictor,
    generate_temporal_data,
    TREND_LABELS,
)
from src.utils.helpers import setup_logging

logger = setup_logging()


def train(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.001,
    n_sequences: int = 1500,
    save_path: str = "checkpoints/temporal_model.pt",
):
    logger.info("=" * 60)
    logger.info("Training Temporal Burnout Predictor (GRU)")
    logger.info("=" * 60)

    # Generate data
    logger.info("Generating %d synthetic temporal sequences...", n_sequences)
    sequences, labels = generate_temporal_data(n_sequences=n_sequences)

    # Split 80/20
    split = int(len(labels) * 0.8)
    train_ds = TensorDataset(sequences[:split], labels[:split])
    val_ds = TensorDataset(sequences[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = TemporalBurnoutPredictor()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output["logits"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            train_correct += (output["predicted_trend"] == batch_y).sum().item()
            train_total += len(batch_y)

        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output["logits"], batch_y)
                val_loss += loss.item() * len(batch_y)
                val_correct += (output["predicted_trend"] == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        scheduler.step(avg_val_loss)

        logger.info(
            "Epoch %2d/%d | Train Acc: %.1f%% | Val Acc: %.1f%%",
            epoch + 1, epochs, train_acc * 100, val_acc * 100,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best model saved (val_acc=%.1f%%)", val_acc * 100)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    logger.info("=" * 60)
    logger.info("Training complete! Best val accuracy: %.1f%%", best_val_acc * 100)
    logger.info("Model saved to: %s", save_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal GRU Model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sequences", type=int, default=1500)
    parser.add_argument("--save", type=str, default="checkpoints/temporal_model.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_sequences=args.sequences,
        save_path=args.save,
    )
