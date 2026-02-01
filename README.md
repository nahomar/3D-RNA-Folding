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

### Custom Transformer

The custom model (`model.py`) uses:
- **Embedding Layer**: Nucleotide embeddings (A, C, G, U)
- **Positional Encoding**: Sinusoidal encoding for sequence position
- **Transformer Encoder**: Multi-head self-attention with pre-norm
- **Coordinate Head**: MLP that predicts (x, y, z) for 5 atoms per nucleotide

Default configuration:
- Embedding dimension: 128
- Attention heads: 4
- Transformer layers: 3
- Max sequence length: 512

### RhoFold+

RhoFold+ uses a language model-based approach with:
- RNA-FM embeddings
- E2EFormer architecture
- Structure module for coordinate prediction
- Optional AMBER relaxation

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
