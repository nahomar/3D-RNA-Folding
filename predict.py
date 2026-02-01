"""
Inference script for generating submission predictions.
"""
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path

from config import Config, NUCLEOTIDE_VOCAB
from model import RNATransformer
from data_utils import load_sequences, tokenize_sequence


def load_model(checkpoint_path: Path, config: Config) -> tuple:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Create model
    model = RNATransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    # Get normalization stats
    coord_mean = checkpoint.get('coord_mean', 0.0)
    coord_std = checkpoint.get('coord_std', 1.0)

    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Coord normalization: mean={coord_mean:.4f}, std={coord_std:.4f}")

    return model, coord_mean, coord_std


@torch.no_grad()
def predict_single(
    model: RNATransformer,
    sequence: str,
    coord_mean: float,
    coord_std: float,
    config: Config,
) -> np.ndarray:
    """
    Predict coordinates for a single sequence.

    Returns:
        Array of shape [seq_len, 15] with denormalized coordinates
    """
    # Tokenize
    tokens = tokenize_sequence(sequence)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    # Truncate if needed
    if tokens.shape[1] > config.max_seq_len:
        tokens = tokens[:, :config.max_seq_len]

    # Move to device
    tokens = tokens.to(config.device)
    attention_mask = torch.ones_like(tokens, dtype=torch.float32)

    # Predict
    with autocast(enabled=config.use_amp):
        pred = model(tokens, attention_mask)  # [1, seq_len, 15]

    # Denormalize
    pred = pred.cpu().numpy()[0]  # [seq_len, 15]
    pred = pred * coord_std + coord_mean

    return pred


def generate_submission(
    model: RNATransformer,
    coord_mean: float,
    coord_std: float,
    config: Config,
) -> pd.DataFrame:
    """Generate submission file."""
    # Load test sequences
    test_df = load_sequences(config.test_sequences)
    print(f"Loaded {len(test_df)} test sequences")

    # Load sample submission for format reference
    sample_sub = pd.read_csv(config.sample_submission)
    print(f"Sample submission has {len(sample_sub)} rows")

    # Get column names from sample submission
    coord_cols = [col for col in sample_sub.columns if col.startswith(('x_', 'y_', 'z_'))]
    print(f"Coordinate columns: {coord_cols}")

    # Generate predictions
    all_rows = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        target_id = row['target_id']
        sequence = row['sequence']

        # Predict coordinates
        coords = predict_single(model, sequence, coord_mean, coord_std, config)

        # Create rows for each residue
        for i, nuc in enumerate(sequence):
            if i >= len(coords):
                # Sequence longer than max_seq_len, use zeros for remaining
                coord_row = [0.0] * len(coord_cols)
            else:
                coord_row = coords[i].tolist()
                # Pad if we have fewer coordinates than needed
                while len(coord_row) < len(coord_cols):
                    coord_row.append(0.0)
                coord_row = coord_row[:len(coord_cols)]

            row_data = {
                'ID': f"{target_id}_{i+1}",
                'resname': nuc,
                'resid': i + 1,
            }
            for j, col in enumerate(coord_cols):
                row_data[col] = coord_row[j]

            all_rows.append(row_data)

    # Create submission dataframe
    submission = pd.DataFrame(all_rows)

    # Ensure column order matches sample submission
    submission = submission[sample_sub.columns]

    return submission


def main():
    """Main prediction function."""
    config = Config()

    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
        config.use_amp = False

    print(f"Using device: {config.device}")

    # Load model
    checkpoint_path = config.checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please train the model first by running train.py")
        return

    model, coord_mean, coord_std = load_model(checkpoint_path, config)

    # Generate submission
    print("\nGenerating submission...")
    submission = generate_submission(model, coord_mean, coord_std, config)

    # Save submission
    submission.to_csv(config.submission_path, index=False)
    print(f"\nSubmission saved to {config.submission_path}")
    print(f"Submission shape: {submission.shape}")

    # Print sample
    print("\nFirst few rows:")
    print(submission.head(10))


if __name__ == "__main__":
    main()
