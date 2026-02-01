"""
Data loading, tokenization, and PyTorch Dataset for RNA structure prediction.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import Config, NUCLEOTIDE_VOCAB


def tokenize_sequence(sequence: str) -> List[int]:
    """Convert RNA sequence string to list of token IDs."""
    return [NUCLEOTIDE_VOCAB.get(nuc, 0) for nuc in sequence.upper()]


def load_sequences(path: Path) -> pd.DataFrame:
    """Load sequences CSV file."""
    df = pd.read_csv(path)
    return df


def load_labels(path: Path) -> pd.DataFrame:
    """Load labels CSV file."""
    df = pd.read_csv(path)
    return df


def get_coord_columns(df: pd.DataFrame) -> List[str]:
    """Get coordinate column names from dataframe."""
    coord_cols = []
    for col in df.columns:
        if col.startswith(('x_', 'y_', 'z_')):
            coord_cols.append(col)
    return sorted(coord_cols, key=lambda x: (int(x.split('_')[1]), x.split('_')[0]))


def group_labels_by_target(labels_df: pd.DataFrame, num_coords: int = 5) -> Dict[str, np.ndarray]:
    """
    Group labels by target_id and create coordinate arrays.

    Returns dict mapping target_id to array of shape [seq_len, num_coords * 3]
    """
    # Get coordinate columns
    coord_cols = get_coord_columns(labels_df)

    # Limit to requested number of coordinates (5 atoms * 3 = 15)
    max_cols = num_coords * 3
    if len(coord_cols) > max_cols:
        coord_cols = coord_cols[:max_cols]

    # Group by target (extract target_id from ID column)
    labels_df = labels_df.copy()
    labels_df['target_id'] = labels_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    grouped = {}
    for target_id, group in labels_df.groupby('target_id'):
        # Sort by resid to ensure correct order
        group = group.sort_values('resid')

        # Extract coordinates
        coords = group[coord_cols].values.astype(np.float32)

        # Pad if we have fewer coordinate columns than needed
        if coords.shape[1] < max_cols:
            padding = np.full((coords.shape[0], max_cols - coords.shape[1]), -1e18, dtype=np.float32)
            coords = np.concatenate([coords, padding], axis=1)

        grouped[target_id] = coords

    return grouped


class RNADataset(Dataset):
    """PyTorch Dataset for RNA sequences and their 3D coordinates."""

    def __init__(
        self,
        sequences_df: pd.DataFrame,
        labels_dict: Optional[Dict[str, np.ndarray]] = None,
        config: Config = None,
        is_test: bool = False
    ):
        self.config = config or Config()
        self.is_test = is_test

        # Store sequences
        self.sequences = sequences_df['sequence'].tolist()
        self.target_ids = sequences_df['target_id'].tolist()

        # Store labels if provided
        self.labels_dict = labels_dict

        # Compute normalization statistics from labels (if available)
        self.coord_mean = 0.0
        self.coord_std = 1.0
        if labels_dict and not is_test:
            all_coords = []
            for coords in labels_dict.values():
                valid_mask = coords > -1e17  # Not missing
                all_coords.extend(coords[valid_mask].flatten())
            if all_coords:
                self.coord_mean = np.mean(all_coords)
                self.coord_std = np.std(all_coords) + 1e-8

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        target_id = self.target_ids[idx]

        # Tokenize sequence
        tokens = tokenize_sequence(sequence)
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Truncate if too long
        if len(tokens) > self.config.max_seq_len:
            tokens = tokens[:self.config.max_seq_len]

        result = {
            'tokens': tokens,
            'target_id': target_id,
            'seq_len': len(tokens),
        }

        # Add labels if available
        if self.labels_dict is not None and target_id in self.labels_dict:
            coords = self.labels_dict[target_id]

            # Truncate coords to match sequence length
            if len(coords) > len(tokens):
                coords = coords[:len(tokens)]

            # Create mask for valid coordinates (not -1e18)
            valid_mask = (coords > -1e17).astype(np.float32)

            # Normalize coordinates (only valid ones)
            coords_normalized = coords.copy()
            coords_normalized = np.where(
                coords > -1e17,
                (coords - self.coord_mean) / self.coord_std,
                0.0  # Set missing to 0 after normalization
            )

            result['coords'] = torch.tensor(coords_normalized, dtype=torch.float32)
            result['coord_mask'] = torch.tensor(valid_mask, dtype=torch.float32)

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for padding variable-length sequences."""
    # Pad tokens
    tokens = [item['tokens'] for item in batch]
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros_like(tokens_padded, dtype=torch.float32)
    for i, t in enumerate(tokens):
        attention_mask[i, :len(t)] = 1.0

    result = {
        'tokens': tokens_padded,
        'attention_mask': attention_mask,
        'target_ids': [item['target_id'] for item in batch],
        'seq_lens': torch.tensor([item['seq_len'] for item in batch]),
    }

    # Pad coordinates if available
    if 'coords' in batch[0]:
        coords = [item['coords'] for item in batch]
        coord_masks = [item['coord_mask'] for item in batch]

        # Pad to max length in batch
        max_len = tokens_padded.shape[1]
        coord_dim = coords[0].shape[1]

        coords_padded = torch.zeros(len(batch), max_len, coord_dim)
        masks_padded = torch.zeros(len(batch), max_len, coord_dim)

        for i, (c, m) in enumerate(zip(coords, coord_masks)):
            seq_len = min(len(c), max_len)
            coords_padded[i, :seq_len] = c[:seq_len]
            masks_padded[i, :seq_len] = m[:seq_len]

        result['coords'] = coords_padded
        result['coord_mask'] = masks_padded

    return result


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    # Load data
    print("Loading sequences...")
    train_seq_df = load_sequences(config.train_sequences)
    val_seq_df = load_sequences(config.val_sequences)
    test_seq_df = load_sequences(config.test_sequences)

    print("Loading labels...")
    train_labels_df = load_labels(config.train_labels)
    val_labels_df = load_labels(config.val_labels)

    print("Grouping labels by target...")
    train_labels = group_labels_by_target(train_labels_df, num_coords=config.num_atoms)
    val_labels = group_labels_by_target(val_labels_df, num_coords=config.num_atoms)

    print(f"Train sequences: {len(train_seq_df)}, Train targets with labels: {len(train_labels)}")
    print(f"Val sequences: {len(val_seq_df)}, Val targets with labels: {len(val_labels)}")
    print(f"Test sequences: {len(test_seq_df)}")

    # Create datasets
    train_dataset = RNADataset(train_seq_df, train_labels, config)
    val_dataset = RNADataset(val_seq_df, val_labels, config)
    val_dataset.coord_mean = train_dataset.coord_mean
    val_dataset.coord_std = train_dataset.coord_std

    test_dataset = RNADataset(test_seq_df, None, config, is_test=True)
    test_dataset.coord_mean = train_dataset.coord_mean
    test_dataset.coord_std = train_dataset.coord_std

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Store normalization stats for later use
    test_loader.coord_mean = train_dataset.coord_mean
    test_loader.coord_std = train_dataset.coord_std

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    config = Config()
    train_loader, val_loader, test_loader = create_dataloaders(config)

    print("\nTesting train loader...")
    for batch in train_loader:
        print(f"Tokens shape: {batch['tokens'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        if 'coords' in batch:
            print(f"Coords shape: {batch['coords'].shape}")
            print(f"Coord mask shape: {batch['coord_mask'].shape}")
        break

    print("\nData loading test complete!")
