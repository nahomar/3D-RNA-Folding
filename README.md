# 3D RNA Folding

A deep learning project for predicting 3D structures of RNA molecules. This repository contains a custom Transformer-based model for RNA structure prediction, along with integration of [RhoFold+](https://github.com/ml4bio/RhoFold), a state-of-the-art RNA 3D structure prediction method.

## Overview

RNA 3D structure prediction is crucial for understanding RNA function and designing RNA-based therapeutics. This project provides:

- **Custom Transformer Model**: A lightweight transformer architecture that predicts 3D coordinates (x, y, z) for each nucleotide's atoms directly from RNA sequences
- **RhoFold+ Integration**: Pre-integrated RhoFold+ for high-accuracy predictions using language model-based deep learning
- **Kaggle Pipeline**: Ready-to-use notebooks for the Stanford RNA 3D Folding competition

## Project Structure

```
├── train.py           # Training script with mixed precision support
├── predict.py         # Inference and submission generation
├── model.py           # Transformer architecture for structure prediction
├── config.py          # Hyperparameters and configuration
├── data_utils.py      # Data loading and preprocessing utilities
├── rhofold/           # RhoFold+ integration
│   ├── inference.py   # RhoFold+ inference script
│   ├── kaggle_notebooks/  # Competition notebooks
│   └── rhofold/       # Core RhoFold+ model
└── outputs/           # Model checkpoints and predictions
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/nahomar/3D-RNA-Folding.git
cd 3D-RNA-Folding

# Create conda environment (recommended)
conda create -n rna-folding python=3.10
conda activate rna-folding

# Install dependencies
pip install torch numpy pandas tqdm
```

### For RhoFold+ (Optional)

```bash
# Install RhoFold+ dependencies
conda env create -f rhofold/envs/environment_linux.yaml
conda activate RhoFold

# Download pretrained model
cd rhofold/pretrained
wget https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt -O RhoFold_pretrained.pt
```

## Usage

### Training the Custom Model

```bash
python train.py
```

The training script supports:
- Mixed precision training (AMP)
- Cosine annealing learning rate with warmup
- Early stopping
- Automatic checkpointing

### Making Predictions

```bash
python predict.py
```

This generates a submission file with predicted 3D coordinates for test sequences.

### Using RhoFold+

```bash
cd rhofold

# With sequence and MSA
python inference.py \
    --input_fas ./example/input/3owzA/3owzA.fasta \
    --input_a3m ./example/input/3owzA/3owzA.a3m \
    --output_dir ./output/3owzA/ \
    --ckpt ./pretrained/RhoFold_pretrained.pt

# Single sequence (faster but less accurate)
python inference.py \
    --input_fas ./example/input/3owzA/3owzA.fasta \
    --single_seq_pred True \
    --output_dir ./output/3owzA/
```

## Model Architecture

### RhoFold+ (State-of-the-Art)

RhoFold+ (`rhofold/`) is a language model-based deep learning approach:

```
Input Sequence → RNA-FM (12 layers) → Rhoformer (12 layers) → Structure Module (8 layers) → 3D Coordinates
      ↑                                      ↑
     MSA ←──────────────────────── Recycling x 10 ──────────────────────────┘
```

Components:
- **RNA-FM**: Pre-trained RNA language model for sequence embeddings
- **Rhoformer/E2EFormer**: MSA attention + Pair representations + Triangle attention
- **Structure Module**: Invariant Point Attention (IPA) for 3D coordinate prediction
- **Recycling**: Iterative refinement (10 cycles)
- **AMBER Relaxation**: Optional molecular dynamics refinement

### Custom Transformer (Lightweight)

The custom model (`model.py`) is a simpler, faster alternative:
- Nucleotide embeddings (A, C, G, U)
- Transformer encoder (3 layers, 4 heads)
- MLP coordinate prediction head

## Advanced Usage: Leveraging RhoFold+

### Option 1: Direct Inference with RhoFold+

```python
from rhofold_enhanced_model import RhoFoldWrapper

wrapper = RhoFoldWrapper(device='cuda')
coords, plddt, secondary_structure = wrapper.predict("GGGAAACCC")
```

### Option 2: Knowledge Distillation

Train a smaller, faster model using RhoFold+ as a teacher:

```bash
python train_rhofold.py --mode distill --epochs 50 --device cuda
```

This creates a lightweight student model (~10M params) that learns from RhoFold's predictions (~100M params).

### Option 3: Hybrid Model with Frozen RhoFold Backbone

Use RhoFold's powerful feature extraction with custom prediction heads:

```python
from rhofold_enhanced_model import HybridRNAModel

model = HybridRNAModel(freeze_backbone=True)
# Only custom heads are trainable, RhoFold features are frozen
output = model(tokens, rna_fm_tokens)
```

### Option 4: Pre-generate RhoFold Predictions

Generate predictions for your dataset to use as pseudo-labels:

```bash
python train_rhofold.py --mode generate
```

### Key Files for RhoFold Integration

| File | Purpose |
|------|---------|
| `rhofold_enhanced_model.py` | Wrapper classes and hybrid models |
| `train_rhofold.py` | Training scripts for distillation/fine-tuning |
| `rhofold/inference.py` | Direct RhoFold+ inference |
| `rhofold/rhofold/rhofold.py` | Core RhoFold+ model |

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class Config:
    # Model architecture
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 50
    
    # Device
    device: str = "cuda"  # or "mps" for Apple Silicon
```

## Output Format

Predictions are saved as CSV with columns:
- `ID`: Target ID and residue number
- `resname`: Nucleotide (A, C, G, U)
- `resid`: Residue index
- `x_1, y_1, z_1, ..., x_5, y_5, z_5`: 3D coordinates for 5 atoms

## Citations

If you use RhoFold+ in your research, please cite:

```bibtex
@article{shen2024accurate,
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  journal={Nature Methods},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](rhofold/LICENSE) for details.

## Acknowledgments

- [RhoFold+](https://github.com/ml4bio/RhoFold) by the CUHK AI Health Lab
- Stanford RNA 3D Folding Kaggle Competition
