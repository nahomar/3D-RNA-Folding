"""
Enhanced RNA Folding Model using RhoFold+ components.

This module provides several approaches to leverage RhoFold+ architecture:
1. Use RNA-FM embeddings as features
2. Use full RhoFold+ with fine-tuning
3. Extract RhoFold predictions as pseudo-labels
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys

# Add rhofold to path
sys.path.insert(0, str(Path(__file__).parent / "rhofold"))

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features


class RhoFoldWrapper:
    """
    Wrapper for using RhoFold+ for inference.
    
    Usage:
        wrapper = RhoFoldWrapper(device='cuda:0')
        coords, plddt, secondary_structure = wrapper.predict("ACGUACGU")
    """
    
    def __init__(
        self, 
        checkpoint_path: str = "./rhofold/pretrained/rhofold_pretrained_params.pt",
        device: str = "cuda"
    ):
        self.device = device
        self.model = RhoFold(rhofold_config)
        
        # Load pretrained weights
        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            self.model.load_state_dict(state_dict)
            print(f"Loaded RhoFold weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Download from: https://huggingface.co/cuhkaih/rhofold")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(
        self, 
        sequence: str,
        msa_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict 3D structure for an RNA sequence.
        
        Args:
            sequence: RNA sequence (A, U, G, C)
            msa_path: Optional path to MSA file (.a3m)
            
        Returns:
            coords: [seq_len, num_atoms, 3] 3D coordinates
            plddt: [seq_len] confidence scores (0-1)
            ss_prob: [seq_len, seq_len] secondary structure probability
        """
        # Create temporary fasta file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">query\n{sequence}\n")
            fasta_path = f.name
        
        # Get features
        if msa_path is None:
            msa_path = fasta_path  # Single sequence mode
            
        data_dict = get_features(fasta_path, msa_path)
        
        # Forward pass
        outputs = self.model(
            tokens=data_dict['tokens'].to(self.device),
            rna_fm_tokens=data_dict['rna_fm_tokens'].to(self.device),
            seq=data_dict['seq'],
        )
        
        output = outputs[-1]  # Get last recycling iteration
        
        # Extract predictions
        coords = output['cord_tns_pred'][-1].squeeze(0).cpu().numpy()
        plddt = output['plddt'][0].cpu().numpy()
        ss_prob = torch.sigmoid(output['ss'][0, 0]).cpu().numpy()
        
        # Clean up
        Path(fasta_path).unlink()
        
        return coords, plddt, ss_prob
    
    def predict_batch(
        self, 
        sequences: list,
        batch_size: int = 4
    ) -> list:
        """Predict structures for multiple sequences."""
        results = []
        for seq in sequences:
            coords, plddt, ss = self.predict(seq)
            results.append({
                'coords': coords,
                'plddt': plddt,
                'secondary_structure': ss
            })
        return results


class RhoFoldFeatureExtractor(nn.Module):
    """
    Extract features from RhoFold+ for use in custom models.
    
    This extracts:
    - RNA-FM embeddings (language model features)
    - MSA embeddings
    - Pair representations
    
    These can be used as input features for your own downstream model.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "./rhofold/pretrained/rhofold_pretrained_params.pt",
        freeze_rhofold: bool = True
    ):
        super().__init__()
        
        self.rhofold = RhoFold(rhofold_config)
        
        # Load pretrained weights
        if Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            self.rhofold.load_state_dict(state_dict)
        
        # Optionally freeze RhoFold weights
        if freeze_rhofold:
            for param in self.rhofold.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        tokens: torch.Tensor,
        rna_fm_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from RhoFold.
        
        Returns:
            Dictionary with:
            - 'msa_features': [batch, msa_depth, seq_len, c_m]
            - 'pair_features': [batch, seq_len, seq_len, c_z]
            - 'single_features': [batch, seq_len, c_s]
        """
        msa_tokens = tokens[:, :rhofold_config.globals.msa_depth]
        
        # Get embeddings
        msa_fea, pair_fea = self.rhofold.msa_embedder(
            tokens=msa_tokens,
            rna_fm_tokens=rna_fm_tokens,
            is_BKL=True
        )
        
        # Run through E2Eformer (one pass, no recycling)
        device = tokens.device
        msa_fea, pair_fea, single_fea = self.rhofold.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=torch.ones(msa_fea.shape[:3]).to(device),
            pair_mask=torch.ones(pair_fea.shape[:3]).to(device),
            chunk_size=None,
        )
        
        return {
            'msa_features': msa_fea,
            'pair_features': pair_fea,
            'single_features': single_fea,
        }


class HybridRNAModel(nn.Module):
    """
    Hybrid model that combines RhoFold features with custom prediction heads.
    
    This allows you to:
    1. Leverage RhoFold's pretrained representations
    2. Add your own task-specific heads
    3. Fine-tune on your specific data
    """
    
    def __init__(
        self,
        checkpoint_path: str = "./rhofold/pretrained/rhofold_pretrained_params.pt",
        freeze_backbone: bool = True,
        num_atoms: int = 5,
    ):
        super().__init__()
        
        # RhoFold feature extractor
        self.feature_extractor = RhoFoldFeatureExtractor(
            checkpoint_path=checkpoint_path,
            freeze_rhofold=freeze_backbone
        )
        
        # Custom coordinate prediction head
        c_s = rhofold_config.globals.c_s  # 384
        
        self.coord_head = nn.Sequential(
            nn.Linear(c_s, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_atoms * 3),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(c_s, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        rna_fm_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tokens: MSA tokens [batch, msa_depth, seq_len]
            rna_fm_tokens: RNA-FM tokens [batch, seq_len]
            
        Returns:
            coords: [batch, seq_len, num_atoms * 3]
            confidence: [batch, seq_len, 1]
        """
        # Extract features
        features = self.feature_extractor(tokens, rna_fm_tokens)
        single_fea = features['single_features']  # [batch, seq_len, c_s]
        
        # Predict coordinates
        coords = self.coord_head(single_fea)
        
        # Predict confidence
        confidence = self.confidence_head(single_fea)
        
        return {
            'coords': coords,
            'confidence': confidence,
            'features': features,
        }


class RhoFoldDistillation(nn.Module):
    """
    Use RhoFold+ as a teacher to train a smaller student model.
    
    Benefits:
    - Smaller, faster model for deployment
    - Can work without MSA at inference time
    - Knowledge distillation from RhoFold's predictions
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_atoms: int = 5,
    ):
        super().__init__()
        
        # Vocabulary: PAD, A, C, G, U
        self.embedding = nn.Embedding(5, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        
        # Transformer encoder (student)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, num_atoms * 3),
        )
        
        # pLDDT prediction head (confidence)
        self.plddt_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 50),  # 50 bins like RhoFold
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: [batch, seq_len] nucleotide tokens
            attention_mask: [batch, seq_len] mask (1=valid, 0=pad)
        """
        # Embed
        x = self.embedding(tokens)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Predict
        coords = self.coord_head(x)
        plddt_logits = self.plddt_head(x)
        
        return {
            'coords': coords,
            'plddt_logits': plddt_logits,
            'features': x,
        }


def compute_distillation_loss(
    student_output: Dict[str, torch.Tensor],
    teacher_output: Dict[str, torch.Tensor],
    coord_weight: float = 1.0,
    plddt_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute distillation loss between student and teacher (RhoFold).
    
    Args:
        student_output: Output from student model
        teacher_output: Output from RhoFold teacher
        coord_weight: Weight for coordinate loss
        plddt_weight: Weight for confidence loss
    """
    # Coordinate MSE loss
    coord_loss = nn.functional.mse_loss(
        student_output['coords'],
        teacher_output['coords'].detach()
    )
    
    # pLDDT cross-entropy loss
    plddt_loss = nn.functional.cross_entropy(
        student_output['plddt_logits'].view(-1, 50),
        teacher_output['plddt_bins'].view(-1).detach()
    )
    
    return coord_weight * coord_loss + plddt_weight * plddt_loss


# ============================================================
# Example usage and training script
# ============================================================

def example_inference():
    """Example: Use RhoFold for inference."""
    print("=" * 60)
    print("Example 1: Direct RhoFold Inference")
    print("=" * 60)
    
    wrapper = RhoFoldWrapper(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    sequence = "GGGAAACCC"  # Simple hairpin
    coords, plddt, ss = wrapper.predict(sequence)
    
    print(f"Sequence: {sequence}")
    print(f"Coordinates shape: {coords.shape}")
    print(f"Average pLDDT: {plddt.mean():.3f}")
    print(f"Secondary structure probability shape: {ss.shape}")


def example_feature_extraction():
    """Example: Extract RhoFold features for custom model."""
    print("\n" + "=" * 60)
    print("Example 2: Feature Extraction")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create hybrid model
    model = HybridRNAModel(freeze_backbone=True).to(device)
    
    print(f"Hybrid model created")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_distillation():
    """Example: Train a student model with RhoFold as teacher."""
    print("\n" + "=" * 60)
    print("Example 3: Knowledge Distillation")
    print("=" * 60)
    
    # Create student model (much smaller than RhoFold)
    student = RhoFoldDistillation(
        embed_dim=256,
        num_layers=6,
        num_heads=8,
    )
    
    print(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
    print("This is much smaller than RhoFold (~100M params)")
    print("\nTraining loop would:")
    print("1. Generate predictions from RhoFold teacher")
    print("2. Train student to match teacher's predictions")
    print("3. Result: Fast model that doesn't need MSA")


if __name__ == "__main__":
    print("RhoFold+ Enhanced Model Options\n")
    
    # Check if RhoFold checkpoint exists
    ckpt_path = Path("./rhofold/pretrained/rhofold_pretrained_params.pt")
    if not ckpt_path.exists():
        print("Note: RhoFold checkpoint not found.")
        print(f"Download from: https://huggingface.co/cuhkaih/rhofold")
        print(f"Save to: {ckpt_path}\n")
    
    # Run examples
    example_feature_extraction()
    example_distillation()
    
    if ckpt_path.exists():
        example_inference()
