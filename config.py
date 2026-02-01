"""
Configuration and hyperparameters for RNA 3D structure prediction.
"""
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data paths
    data_dir: Path = Path("/Users/nahom/Downloads")
    train_sequences: Path = data_dir / "train_sequences.csv"
    train_labels: Path = data_dir / "train_labels.csv"
    val_sequences: Path = data_dir / "validation_sequences.csv"
    val_labels: Path = data_dir / "validation_labels.csv"
    test_sequences: Path = data_dir / "test_sequences.csv"
    sample_submission: Path = data_dir / "sample_submission.csv"

    # Output paths
    output_dir: Path = Path("/Users/nahom/rna_folding/outputs")
    checkpoint_dir: Path = output_dir / "checkpoints"
    submission_path: Path = output_dir / "submission.csv"

    # Model architecture (reduced for faster training)
    vocab_size: int = 5  # PAD, A, C, G, U
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512

    # Output dimensions
    num_atoms: int = 5  # Predict 5 atoms per residue
    coord_dim: int = 3  # x, y, z
    output_dim: int = num_atoms * coord_dim  # 15

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    warmup_steps: int = 500
    grad_clip: float = 1.0
    patience: int = 10  # Early stopping patience

    # Mixed precision (not supported on MPS)
    use_amp: bool = False

    # Device
    device: str = "mps"

    # Reproducibility
    seed: int = 42

    # Missing value marker in labels
    missing_value: float = -1e+18

    def __post_init__(self):
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Nucleotide vocabulary
NUCLEOTIDE_VOCAB = {
    '<PAD>': 0,
    'A': 1,
    'C': 2,
    'G': 3,
    'U': 4,
}

# Reverse vocabulary for decoding
IDX_TO_NUCLEOTIDE = {v: k for k, v in NUCLEOTIDE_VOCAB.items()}
