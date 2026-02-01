"""
Transformer-based model for RNA 3D structure prediction.
"""
import math
import torch
import torch.nn as nn
from typing import Optional

from config import Config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        Returns:
            Tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CoordinateHead(nn.Module):
    """MLP head for predicting 3D coordinates."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            Tensor of shape [batch, seq_len, output_dim]
        """
        return self.mlp(x)


class RNATransformer(nn.Module):
    """
    Transformer-based model for RNA 3D structure prediction.

    Takes RNA sequences as input and predicts 3D coordinates for each nucleotide.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Nucleotide embedding
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.embed_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # Coordinate prediction head
        self.coord_head = CoordinateHead(
            input_dim=config.embed_dim,
            hidden_dim=config.ffn_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Input token IDs [batch, seq_len]
            attention_mask: Mask for padding [batch, seq_len], 1 for real tokens, 0 for padding

        Returns:
            Predicted coordinates [batch, seq_len, output_dim]
        """
        # Embed tokens
        x = self.embedding(tokens)  # [batch, seq_len, embed_dim]

        # Scale embeddings
        x = x * math.sqrt(self.config.embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)  # [batch, seq_len, embed_dim]

        # Create attention mask for transformer (True = masked/ignored)
        if attention_mask is not None:
            # Convert from (1=valid, 0=pad) to (True=masked, False=valid)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [batch, seq_len, embed_dim]

        # Predict coordinates
        coords = self.coord_head(x)  # [batch, seq_len, output_dim]

        return coords

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskedMSELoss(nn.Module):
    """MSE loss that ignores masked (missing) values."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predictions [batch, seq_len, dim]
            target: Targets [batch, seq_len, dim]
            mask: Valid mask [batch, seq_len, dim], 1 for valid, 0 for masked

        Returns:
            Scalar loss value
        """
        # Compute squared error
        squared_error = (pred - target) ** 2

        # Apply mask
        masked_error = squared_error * mask

        # Mean over valid elements
        num_valid = mask.sum() + 1e-8
        loss = masked_error.sum() / num_valid

        return loss


def create_model(config: Config) -> RNATransformer:
    """Create and return the model."""
    model = RNATransformer(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    config = Config()
    model = create_model(config)

    # Test forward pass
    batch_size = 4
    seq_len = 100
    tokens = torch.randint(1, 5, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = model(tokens, attention_mask)

    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, {seq_len}, {config.output_dim}]")

    # Test loss
    loss_fn = MaskedMSELoss()
    target = torch.randn(batch_size, seq_len, config.output_dim)
    mask = torch.ones_like(target)
    loss = loss_fn(output, target, mask)
    print(f"Test loss: {loss.item():.4f}")
