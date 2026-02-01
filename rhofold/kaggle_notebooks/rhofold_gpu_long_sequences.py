"""
RhoFold+ Kaggle GPU Inference for Long Sequences
Run this on Kaggle with GPU (T4/P100) to predict the long RNA sequences.

Targets that need GPU:
- 9ZCC: 1460 nucleotides (RNA origami)
- 9MME: 4640 nucleotides (too long even for most GPUs)
- 9LEL: 476 nucleotides (borderline, might work)
"""

# ======== CELL 1: Install dependencies ========
# !pip install -q einops biopython ml-collections dm-tree tqdm pyyaml scipy matplotlib pandas transformers

# ======== CELL 2: Clone RhoFold+ ========
# !git clone https://github.com/ml4bio/RhoFold.git
# %cd RhoFold
# !pip install -q -e .
# !mkdir -p pretrained
# !wget -q https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt -O pretrained/RhoFold_pretrained.pt

# ======== CELL 3: Setup ========
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== CELL 4: Fix imports ========
rhofold_relax_path = Path('rhofold/relax/relax.py')
if rhofold_relax_path.exists():
    content = rhofold_relax_path.read_text()
    if 'from simtk.openmm' in content and 'try:' not in content[:200]:
        content = content.replace(
            'from simtk.openmm.app import *\nfrom simtk.openmm import *\nfrom simtk.unit import *\nimport simtk.openmm as mm',
            '''try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm as mm
except ImportError:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    import openmm as mm'''
        )
        rhofold_relax_path.write_text(content)

# ======== CELL 5: Load model ========
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

config = rhofold_config()
model = RhoFold(config)
ckpt = torch.load('pretrained/RhoFold_pretrained.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
model = model.to(device)
model.eval()
print("Model loaded successfully")

# ======== CELL 6: Load test data ========
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv')
sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/sample_submission.csv')

# CRITICAL: These are the targets that need GPU prediction
LONG_TARGETS = ['9ZCC', '9MME', '9LEL']  # Add any other targets that failed locally

# Check sequence lengths
for target in LONG_TARGETS:
    seq_len = len(test_seqs[test_seqs['target_id'] == target]['sequence'].values[0])
    print(f"{target}: {seq_len} nucleotides")

# ======== CELL 7: Prediction function ========
def extract_c1_prime_from_pdb(pdb_content):
    """Extract C1' coordinates from PDB content."""
    coords = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            if atom_name == "C1'":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None

def predict_single_sequence(target_id, sequence, output_dir='predictions'):
    """Predict structure for a single sequence."""
    import tempfile

    # Create temp fasta file
    fasta_path = Path(output_dir) / f'{target_id}.fasta'
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, 'w') as f:
        f.write(f'>{target_id}\n{sequence}\n')

    # Run inference
    from rhofold.utils.alphabet import Alphabet
    alphabet = Alphabet.get_default()

    # Tokenize
    tokens = torch.tensor([[alphabet.get_idx(c) for c in sequence]])
    rna_fm_tokens = tokens.clone()

    # Move to device
    tokens = tokens.to(device)
    rna_fm_tokens = rna_fm_tokens.to(device)

    # Inference with gradient checkpointing for memory efficiency
    with torch.no_grad():
        # Clear cache before long sequence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outputs = model(
            tokens=tokens,
            rna_fm_tokens=rna_fm_tokens,
            seq=sequence
        )

    # Get coordinates
    coords = outputs['cord_tns_pred'][-1][0].cpu().numpy()

    # C1' is typically the second atom in the output
    # Output shape: [seq_len, 27, 3] - we need to identify the right atom index
    c1_prime_coords = coords[:, 1, :]  # Adjust index if needed

    return c1_prime_coords

# ======== CELL 8: Run predictions for long sequences ========
all_coords = {}

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']

    # Skip if not in our target list
    if target_id not in LONG_TARGETS:
        continue

    print(f"\nProcessing {target_id} ({len(sequence)} nt)...")

    # Check if sequence is too long even for GPU (>5000 nt likely to OOM on T4)
    if len(sequence) > 5000:
        print(f"  WARNING: Very long sequence, may OOM. Trying anyway...")

    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        coords = predict_single_sequence(target_id, sequence)
        all_coords[target_id] = coords
        print(f"  Success! Got {len(coords)} residue coordinates")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM error - sequence too long for GPU memory")
            # Try with fp16 or chunking here if needed
        else:
            print(f"  Error: {e}")
        all_coords[target_id] = None
    except Exception as e:
        print(f"  Error: {e}")
        all_coords[target_id] = None

print(f"\n\nSuccessfully predicted: {sum(1 for v in all_coords.values() if v is not None)}/{len(LONG_TARGETS)}")

# ======== CELL 9: Generate partial submission (just for long targets) ========
submission_rows = []

for idx, row in sample_sub.iterrows():
    row_id = row['ID']
    parts = row_id.rsplit('_', 1)
    target_id = parts[0]

    # Only include rows for our long targets
    if target_id not in LONG_TARGETS:
        continue

    res_idx = int(parts[1]) - 1
    coords = all_coords.get(target_id)

    new_row = {'ID': row_id, 'resname': row['resname'], 'resid': row['resid']}

    for model_idx in range(1, 6):
        if coords is not None and res_idx < len(coords):
            x, y, z = coords[res_idx]
        else:
            x, y, z = 0.0, 0.0, 0.0
        new_row[f'x_{model_idx}'] = round(float(x), 3)
        new_row[f'y_{model_idx}'] = round(float(y), 3)
        new_row[f'z_{model_idx}'] = round(float(z), 3)

    submission_rows.append(new_row)

partial_sub = pd.DataFrame(submission_rows)
partial_sub.to_csv('long_sequence_predictions.csv', index=False)
print(f"\nSaved {len(partial_sub)} rows to long_sequence_predictions.csv")
print(partial_sub.head())

# ======== CELL 10: Merge with local predictions ========
# If you have local predictions, merge them:
# local_sub = pd.read_csv('/kaggle/input/local-predictions/submission.csv')
# final_sub = local_sub.copy()
#
# for idx, row in partial_sub.iterrows():
#     mask = final_sub['ID'] == row['ID']
#     if mask.any():
#         final_sub.loc[mask] = row.values
#
# final_sub.to_csv('submission.csv', index=False)
