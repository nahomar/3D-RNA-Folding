"""
Training script for RNA 3D structure prediction using RhoFold+ components.

Three training modes:
1. Fine-tune full RhoFold+ on your data
2. Train custom heads on frozen RhoFold features
3. Knowledge distillation to smaller student model
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add rhofold to path
sys.path.insert(0, str(Path(__file__).parent / "rhofold"))

from rhofold_enhanced_model import (
    RhoFoldWrapper,
    HybridRNAModel,
    RhoFoldDistillation,
)
from config import Config, NUCLEOTIDE_VOCAB
from data_utils import create_dataloaders


def train_with_rhofold_teacher(
    student_model: nn.Module,
    teacher: RhoFoldWrapper,
    train_loader,
    val_loader,
    config,
    num_epochs: int = 50,
):
    """
    Train student model using RhoFold as teacher (knowledge distillation).
    
    This creates pseudo-labels from RhoFold predictions and trains a smaller,
    faster model to match them.
    """
    device = config.device
    student_model = student_model.to(device)
    
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("Training with RhoFold Teacher (Knowledge Distillation)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training
        student_model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            tokens = batch['tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get teacher predictions (RhoFold)
            # Note: In practice, you'd pre-compute these for efficiency
            with torch.no_grad():
                sequences = batch.get('sequences', None)
                if sequences:
                    teacher_coords = []
                    for seq in sequences:
                        coords, _, _ = teacher.predict(seq)
                        teacher_coords.append(coords)
                    teacher_coords = torch.tensor(np.array(teacher_coords)).to(device)
                else:
                    # Use ground truth as fallback
                    teacher_coords = batch['coords'].to(device)
            
            # Student forward pass
            optimizer.zero_grad()
            output = student_model(tokens, attention_mask)
            
            # Reshape for loss computation
            pred_coords = output['coords']
            target_coords = teacher_coords[:, :pred_coords.shape[1], :pred_coords.shape[2]]
            
            # MSE loss
            loss = nn.functional.mse_loss(pred_coords, target_coords)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        train_loss /= num_batches
        
        # Validation
        student_model.eval()
        val_loss = 0.0
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_coords = batch['coords'].to(device)
                
                output = student_model(tokens, attention_mask)
                pred_coords = output['coords']
                
                # Match dimensions
                min_len = min(pred_coords.shape[1], target_coords.shape[1])
                min_dim = min(pred_coords.shape[2], target_coords.shape[2])
                
                loss = nn.functional.mse_loss(
                    pred_coords[:, :min_len, :min_dim],
                    target_coords[:, :min_len, :min_dim]
                )
                val_loss += loss.item()
                num_val += 1
        
        val_loss /= max(num_val, 1)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, config.checkpoint_dir / 'student_best.pt')
            print(f"  Saved best model!")
    
    return student_model


def train_hybrid_model(
    model: HybridRNAModel,
    train_loader,
    val_loader,
    config,
    num_epochs: int = 50,
):
    """
    Train hybrid model with frozen RhoFold backbone.
    
    Only the custom heads are trained while RhoFold features remain frozen.
    """
    device = config.device
    model = model.to(device)
    
    # Only optimize trainable parameters (custom heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    optimizer = AdamW(trainable_params, lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("\n" + "=" * 60)
    print("Training Hybrid Model (Frozen RhoFold + Custom Heads)")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Note: Hybrid model needs RhoFold-style inputs (tokens + rna_fm_tokens)
            # This requires preprocessing with get_features()
            tokens = batch['tokens'].to(device)
            coords = batch['coords'].to(device)
            coord_mask = batch['coord_mask'].to(device)
            
            optimizer.zero_grad()
            
            # For hybrid model, you need to prepare rna_fm_tokens
            # This is a simplified version - in practice, use get_features()
            output = model(
                tokens=tokens.unsqueeze(1).expand(-1, 128, -1),  # Fake MSA
                rna_fm_tokens=tokens,
            )
            
            pred = output['coords']
            
            # Masked MSE loss
            min_len = min(pred.shape[1], coords.shape[1])
            min_dim = min(pred.shape[2], coords.shape[2])
            
            mask = coord_mask[:, :min_len, :min_dim]
            loss = ((pred[:, :min_len, :min_dim] - coords[:, :min_len, :min_dim]) ** 2 * mask).sum()
            loss = loss / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        train_loss /= num_batches
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, config.checkpoint_dir / f'hybrid_epoch{epoch+1}.pt')
    
    return model


def generate_rhofold_predictions(
    sequences: list,
    output_dir: Path,
    device: str = 'cuda',
):
    """
    Pre-generate RhoFold predictions for all sequences.
    
    This is useful for:
    1. Creating pseudo-labels for distillation
    2. Ensemble with your custom model
    3. Analyzing RhoFold's predictions
    """
    wrapper = RhoFoldWrapper(device=device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating RhoFold predictions for {len(sequences)} sequences...")
    
    for i, seq in enumerate(tqdm(sequences)):
        try:
            coords, plddt, ss = wrapper.predict(seq)
            
            np.savez(
                output_dir / f"seq_{i}.npz",
                sequence=seq,
                coords=coords,
                plddt=plddt,
                secondary_structure=ss,
            )
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
    
    print(f"Saved predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train RNA folding models with RhoFold")
    parser.add_argument(
        "--mode", 
        choices=['distill', 'hybrid', 'generate'],
        default='distill',
        help="Training mode: distill (knowledge distillation), hybrid (frozen backbone), generate (pre-compute RhoFold predictions)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    config = Config()
    config.device = args.device if torch.cuda.is_available() else 'cpu'
    config.batch_size = args.batch_size
    
    print(f"Device: {config.device}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'generate':
        # Generate RhoFold predictions for training data
        # Load your sequences here
        sequences = ["GGGAAACCC", "AAAUUUGGG"]  # Example
        generate_rhofold_predictions(
            sequences,
            output_dir=config.output_dir / "rhofold_predictions",
            device=config.device
        )
        
    elif args.mode == 'distill':
        # Knowledge distillation
        print("\nCreating dataloaders...")
        train_loader, val_loader, _ = create_dataloaders(config)
        
        print("\nCreating student model...")
        student = RhoFoldDistillation(
            embed_dim=256,
            num_layers=6,
            num_heads=8,
        )
        print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
        
        print("\nLoading RhoFold teacher...")
        teacher = RhoFoldWrapper(device=config.device)
        
        train_with_rhofold_teacher(
            student, teacher, train_loader, val_loader, config,
            num_epochs=args.epochs
        )
        
    elif args.mode == 'hybrid':
        # Hybrid model with frozen backbone
        print("\nCreating dataloaders...")
        train_loader, val_loader, _ = create_dataloaders(config)
        
        print("\nCreating hybrid model...")
        model = HybridRNAModel(freeze_backbone=True)
        
        train_hybrid_model(
            model, train_loader, val_loader, config,
            num_epochs=args.epochs
        )


if __name__ == "__main__":
    main()
