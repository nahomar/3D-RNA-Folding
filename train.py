"""
Training script for RNA 3D structure prediction.
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from config import Config
from model import RNATransformer, MaskedMSELoss, create_model
from data_utils import create_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute Mean Absolute Error on valid elements."""
    abs_error = torch.abs(pred - target) * mask
    num_valid = mask.sum() + 1e-8
    mae = (abs_error.sum() / num_valid).item()
    return mae


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    scaler,
    config: Config,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        # Move to device
        tokens = batch['tokens'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        coords = batch['coords'].to(config.device)
        coord_mask = batch['coord_mask'].to(config.device)

        # Forward pass with mixed precision
        optimizer.zero_grad()
        with autocast(device_type=config.device if config.device in ['cuda', 'mps'] else 'cpu', enabled=config.use_amp):
            pred = model(tokens, attention_mask)
            loss = loss_fn(pred, coords, coord_mask)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            mae = compute_mae(pred, coords, coord_mask)
            total_mae += mae

        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.4f}'})

    return {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn,
    config: Config,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        # Move to device
        tokens = batch['tokens'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        coords = batch['coords'].to(config.device)
        coord_mask = batch['coord_mask'].to(config.device)

        # Forward pass
        with autocast(device_type=config.device if config.device in ['cuda', 'mps'] else 'cpu', enabled=config.use_amp):
            pred = model(tokens, attention_mask)
            loss = loss_fn(pred, coords, coord_mask)

        # Metrics
        total_loss += loss.item()
        mae = compute_mae(pred, coords, coord_mask)
        total_mae += mae
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.4f}'})

    return {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    config: Config,
    coord_mean: float,
    coord_std: float,
    filename: str = "checkpoint.pt",
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config,
        'coord_mean': coord_mean,
        'coord_std': coord_std,
    }
    path = config.checkpoint_dir / filename
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    config: Config,
    filename: str = "checkpoint.pt",
) -> dict:
    """Load model checkpoint."""
    path = config.checkpoint_dir / filename
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def train(config: Config = None):
    """Main training function."""
    if config is None:
        config = Config()

    # Set seed
    set_seed(config.seed)

    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
        config.use_amp = False
    elif config.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        config.device = "cpu"
        config.use_amp = False

    print(f"Using device: {config.device}")
    print(f"Mixed precision: {config.use_amp}")

    # Create dataloaders
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Get normalization stats
    coord_mean = train_loader.dataset.coord_mean
    coord_std = train_loader.dataset.coord_std
    print(f"Coordinate normalization: mean={coord_mean:.4f}, std={coord_std:.4f}")

    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    model = create_model(config)
    model = model.to(config.device)

    # Loss function
    loss_fn = MaskedMSELoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * len(train_loader) - config.warmup_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps],
    )

    # Mixed precision scaler
    scaler = GradScaler(device=config.device if config.device == 'cuda' else 'cpu', enabled=config.use_amp)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, config, epoch
        )

        # Update scheduler (per epoch for simplicity)
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, config)

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.6f}, MAE: {train_metrics['mae']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.6f}, MAE: {val_metrics['mae']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, config,
                coord_mean, coord_std, "best_model.pt"
            )
            print(f"  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, config,
                coord_mean, coord_std, f"checkpoint_epoch{epoch+1}.pt"
            )

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model


if __name__ == "__main__":
    train()
