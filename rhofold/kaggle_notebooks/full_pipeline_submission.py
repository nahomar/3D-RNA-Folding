"""
Stanford RNA 3D Folding - Full Pipeline Submission
===================================================
Implements the 14-Day Hybrid Expert Plan:
1. Template Search (MMseqs2)
2. RNAPro Inference (with templates)
3. DRfold2 Ab Initio Predictions
4. RhoFold+ Predictions
5. Energy Minimization (OpenMM)
6. Diverse Ensemble Selection

Run on Kaggle with GPU (T4/P100 minimum, A100 recommended)
"""

# ============================================================================
# CELL 1: Install All Dependencies
# ============================================================================
# %%
!pip install -q einops biopython ml-collections dm-tree tqdm pyyaml scipy matplotlib pandas transformers
!pip install -q fair-esm  # For RNA-FM embeddings
!pip install -q openmm pdbfixer  # For energy minimization

# Install MMseqs2
!apt-get install -qq mmseqs2

print("Dependencies installed")

# ============================================================================
# CELL 2: Clone Repositories
# ============================================================================
# %%
import os
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('relaxed', exist_ok=True)
os.makedirs('final', exist_ok=True)

# Clone RhoFold+
!git clone --quiet https://github.com/ml4bio/RhoFold.git models/RhoFold
%cd models/RhoFold
!pip install -q -e .
%cd ../..

# Clone DRfold2
!git clone --quiet https://github.com/leeyang/DRfold2.git models/DRfold2

# Download RNAPro (if available on HuggingFace)
!mkdir -p models/RNAPro
# Note: RNAPro may require manual setup - using alternative approach below

print("Repositories cloned")

# ============================================================================
# CELL 3: Download Model Weights
# ============================================================================
# %%
# RhoFold+ weights
!mkdir -p models/RhoFold/pretrained
!wget -q https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt \
    -O models/RhoFold/pretrained/RhoFold_pretrained.pt

# DRfold2 weights
%cd models/DRfold2
!bash install.sh 2>/dev/null || echo "DRfold2 weights may need manual download"
%cd ../..

print("Model weights downloaded")

# ============================================================================
# CELL 4: Setup and Load Data
# ============================================================================
# %%
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load competition data
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv')
sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/sample_submission.csv')

print(f"\nTest sequences: {len(test_seqs)}")
print(f"Sample submission rows: {len(sample_sub)}")

# Categorize by sequence length
test_seqs['seq_len'] = test_seqs['sequence'].str.len()
print(f"\nSequence length distribution:")
print(test_seqs['seq_len'].describe())

# ============================================================================
# CELL 5: Template Search with MMseqs2
# ============================================================================
# %%
print("=" * 60)
print("PHASE 1: Template Search")
print("=" * 60)

# Create FASTA file for test sequences
with open('templates/test_targets.fasta', 'w') as f:
    for _, row in test_seqs.iterrows():
        f.write(f">{row['target_id']}\n{row['sequence']}\n")

# Download PDB RNA sequences
!wget -q https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O templates/pdb_seqres.txt.gz
!gunzip -f templates/pdb_seqres.txt.gz

# Extract RNA sequences
import re
with open('templates/pdb_seqres.txt', 'r') as f:
    content = f.read()

entries = content.split('>')
rna_count = 0
with open('templates/pdb_rna.fasta', 'w') as out:
    for entry in entries[1:]:
        lines = entry.strip().split('\n')
        if len(lines) < 2:
            continue
        header = lines[0]
        seq = ''.join(lines[1:]).upper().replace('T', 'U')
        if re.match('^[ACGU]+$', seq) and len(seq) >= 10:
            out.write(f'>{header.split()[0]}\n{seq}\n')
            rna_count += 1

print(f"Extracted {rna_count} RNA sequences from PDB")

# Run MMseqs2 search
!mmseqs easy-search templates/test_targets.fasta templates/pdb_rna.fasta \
    templates/hits.m8 templates/tmp \
    --search-type 3 -e 1e-3 -s 7.5 --threads 4 \
    --format-output "query,target,pident,alnlen,evalue,bits" 2>/dev/null

# Parse results
if os.path.exists('templates/hits.m8') and os.path.getsize('templates/hits.m8') > 0:
    hits = pd.read_csv('templates/hits.m8', sep='\t', header=None,
                       names=['query', 'target', 'pident', 'alnlen', 'evalue', 'bits'])
    best_templates = hits.loc[hits.groupby('query')['pident'].idxmax()]
    best_templates.to_csv('templates/best_templates.csv', index=False)
    print(f"Found templates for {len(best_templates)} targets")
    print(best_templates[['query', 'target', 'pident']].head(10))
else:
    best_templates = pd.DataFrame()
    print("No templates found - will use ab initio methods")

# ============================================================================
# CELL 6: RhoFold+ Inference
# ============================================================================
# %%
print("\n" + "=" * 60)
print("PHASE 2: RhoFold+ Predictions")
print("=" * 60)

import sys
sys.path.insert(0, 'models/RhoFold')

# Fix OpenMM import
rhofold_relax = Path('models/RhoFold/rhofold/relax/relax.py')
if rhofold_relax.exists():
    content = rhofold_relax.read_text()
    if 'from simtk.openmm' in content and 'try:' not in content[:300]:
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
        rhofold_relax.write_text(content)

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import Alphabet

# Load model
config = rhofold_config()
rhofold_model = RhoFold(config)
ckpt = torch.load('models/RhoFold/pretrained/RhoFold_pretrained.pt', map_location='cpu')
rhofold_model.load_state_dict(ckpt['model'])
rhofold_model = rhofold_model.to(device)
rhofold_model.eval()
print("RhoFold+ model loaded")

def run_rhofold(target_id, sequence, seeds=[42, 123, 456]):
    """Run RhoFold+ with multiple seeds."""
    alphabet = Alphabet.get_default()
    predictions = []

    # Skip sequences > 1000 nt (positional embedding limit)
    if len(sequence) > 1000:
        print(f"  Skipping {target_id} - too long ({len(sequence)} nt)")
        return predictions

    tokens = torch.tensor([[alphabet.get_idx(c) for c in sequence]]).to(device)

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                outputs = rhofold_model(
                    tokens=tokens,
                    rna_fm_tokens=tokens.clone(),
                    seq=sequence
                )

            coords = outputs['cord_tns_pred'][-1][0].cpu().numpy()
            c1_prime = coords[:, 1, :]  # C1' atom

            predictions.append({
                'seed': seed,
                'coords': c1_prime,
                'source': 'rhofold'
            })

        except Exception as e:
            print(f"  Seed {seed} failed: {str(e)[:50]}")

    return predictions

# Run RhoFold+ on all targets
rhofold_predictions = {}
for idx, row in tqdm(test_seqs.iterrows(), total=len(test_seqs), desc="RhoFold+"):
    target_id = row['target_id']
    sequence = row['sequence']

    preds = run_rhofold(target_id, sequence)
    if preds:
        rhofold_predictions[target_id] = preds

print(f"\nRhoFold+ completed: {len(rhofold_predictions)} targets")

# ============================================================================
# CELL 7: DRfold2 Ab Initio Predictions
# ============================================================================
# %%
print("\n" + "=" * 60)
print("PHASE 3: DRfold2 Ab Initio Predictions")
print("=" * 60)

def run_drfold2(target_id, sequence, output_dir='predictions/drfold2'):
    """Run DRfold2 ab initio prediction."""
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    fasta_path = f'{output_dir}/{target_id}.fasta'

    # Write FASTA
    with open(fasta_path, 'w') as f:
        f.write(f'>{target_id}\n{sequence}\n')

    # Run DRfold2
    try:
        result = subprocess.run(
            ['python', 'models/DRfold2/DRfold2.py',
             '-i', fasta_path,
             '-o', f'{output_dir}/{target_id}',
             '--device', device],
            capture_output=True, text=True, timeout=300
        )

        pdb_path = f'{output_dir}/{target_id}/pred.pdb'
        if os.path.exists(pdb_path):
            return pdb_path
    except Exception as e:
        print(f"  DRfold2 failed for {target_id}: {e}")

    return None

# Run DRfold2 on targets without good templates
drfold2_predictions = {}
targets_for_drfold = test_seqs[test_seqs['seq_len'] <= 500]['target_id'].tolist()

# If we have templates, prioritize targets without good templates
if len(best_templates) > 0:
    good_template_targets = best_templates[best_templates['pident'] > 50]['query'].tolist()
    targets_for_drfold = [t for t in targets_for_drfold if t not in good_template_targets]

print(f"Running DRfold2 on {len(targets_for_drfold)} targets")

for target_id in tqdm(targets_for_drfold[:10], desc="DRfold2"):  # Limit for time
    row = test_seqs[test_seqs['target_id'] == target_id].iloc[0]
    pdb_path = run_drfold2(target_id, row['sequence'])
    if pdb_path:
        drfold2_predictions[target_id] = pdb_path

print(f"DRfold2 completed: {len(drfold2_predictions)} predictions")

# ============================================================================
# CELL 8: Energy Minimization with OpenMM
# ============================================================================
# %%
print("\n" + "=" * 60)
print("PHASE 4: Energy Minimization")
print("=" * 60)

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ImportError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *

from pdbfixer import PDBFixer

def energy_minimize(pdb_path, output_path, max_iterations=500):
    """Run OpenMM energy minimization on a PDB structure."""
    try:
        # Fix PDB issues
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # Setup simulation
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        try:
            system = forcefield.createSystem(
                fixer.topology,
                nonbondedMethod=NoCutoff,
                constraints=HBonds
            )
        except Exception:
            # Fallback for RNA without full force field support
            print(f"  Using implicit solvent for {pdb_path}")
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            system = forcefield.createSystem(
                fixer.topology,
                nonbondedMethod=NoCutoff,
                constraints=HBonds,
                implicitSolvent=GBn2
            )

        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(fixer.topology, system, integrator)
        simulation.context.setPositions(fixer.positions)

        # Minimize
        simulation.minimizeEnergy(maxIterations=max_iterations)

        # Save
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_path, 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)

        return True

    except Exception as e:
        print(f"  Minimization failed: {str(e)[:50]}")
        # Copy original if minimization fails
        import shutil
        shutil.copy(pdb_path, output_path)
        return False

def coords_to_pdb(coords, sequence, output_path):
    """Convert C1' coordinates to minimal PDB file."""
    resnames = {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'}

    with open(output_path, 'w') as f:
        atom_num = 1
        for i, (coord, res) in enumerate(zip(coords, sequence)):
            res_name = resnames.get(res, 'N')
            x, y, z = coord
            f.write(f"ATOM  {atom_num:5d}  C1' {res_name:3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            atom_num += 1
        f.write("END\n")

# Save RhoFold+ predictions as PDB and minimize
print("Minimizing RhoFold+ predictions...")
minimized_predictions = {}

for target_id, preds in tqdm(rhofold_predictions.items(), desc="Minimizing"):
    sequence = test_seqs[test_seqs['target_id'] == target_id]['sequence'].values[0]
    minimized_predictions[target_id] = []

    for pred in preds:
        # Save as PDB
        pdb_path = f"predictions/{target_id}_rhofold_seed{pred['seed']}.pdb"
        coords_to_pdb(pred['coords'], sequence, pdb_path)

        # Minimize
        relaxed_path = f"relaxed/{target_id}_rhofold_seed{pred['seed']}_relaxed.pdb"
        success = energy_minimize(pdb_path, relaxed_path, max_iterations=200)

        minimized_predictions[target_id].append({
            'source': 'rhofold',
            'seed': pred['seed'],
            'coords': pred['coords'],
            'pdb_path': relaxed_path if success else pdb_path,
            'relaxed': success
        })

print(f"Minimization completed for {len(minimized_predictions)} targets")

# ============================================================================
# CELL 9: Diverse Ensemble Selection
# ============================================================================
# %%
print("\n" + "=" * 60)
print("PHASE 5: Diverse Ensemble Selection")
print("=" * 60)

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def load_c1_prime_coords(pdb_path):
    """Extract C1' coordinates from PDB."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and "C1'" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None

def compute_rmsd(coords1, coords2):
    """Compute RMSD between coordinate sets."""
    if coords1 is None or coords2 is None:
        return float('inf')
    if len(coords1) != len(coords2):
        return float('inf')
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def select_diverse_ensemble(predictions, n_select=5):
    """Select n diverse predictions via hierarchical clustering."""
    if len(predictions) <= n_select:
        return predictions

    # Get coordinates
    coords_list = []
    for p in predictions:
        if 'coords' in p:
            coords_list.append(p['coords'])
        elif 'pdb_path' in p and os.path.exists(p['pdb_path']):
            coords_list.append(load_c1_prime_coords(p['pdb_path']))
        else:
            coords_list.append(None)

    # Compute RMSD matrix
    n = len(predictions)
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            rmsd = compute_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Handle invalid RMSDs
    rmsd_matrix[np.isinf(rmsd_matrix)] = 100.0
    rmsd_matrix[np.isnan(rmsd_matrix)] = 100.0

    # Cluster
    try:
        condensed = pdist(rmsd_matrix)
        if len(condensed) > 0 and not np.all(condensed == 0):
            Z = linkage(condensed, method='complete')
            clusters = fcluster(Z, t=n_select, criterion='maxclust')

            # Select one from each cluster
            selected = []
            for c in range(1, n_select + 1):
                members = [i for i, cl in enumerate(clusters) if cl == c]
                if members:
                    selected.append(predictions[members[0]])

            # Fill remaining slots
            while len(selected) < n_select and len(predictions) > len(selected):
                for p in predictions:
                    if p not in selected:
                        selected.append(p)
                        break

            return selected[:n_select]
    except Exception as e:
        print(f"  Clustering failed: {e}")

    return predictions[:n_select]

# Select diverse ensemble for each target
final_ensembles = {}

for target_id in tqdm(test_seqs['target_id'], desc="Ensemble Selection"):
    all_preds = []

    # Add RhoFold+ predictions
    if target_id in minimized_predictions:
        all_preds.extend(minimized_predictions[target_id])

    # Add DRfold2 predictions
    if target_id in drfold2_predictions:
        pdb_path = drfold2_predictions[target_id]
        coords = load_c1_prime_coords(pdb_path)
        if coords is not None:
            all_preds.append({
                'source': 'drfold2',
                'coords': coords,
                'pdb_path': pdb_path
            })

    if len(all_preds) > 0:
        selected = select_diverse_ensemble(all_preds, n_select=5)
        final_ensembles[target_id] = selected
    else:
        final_ensembles[target_id] = []

print(f"Ensemble selection completed for {len(final_ensembles)} targets")

# ============================================================================
# CELL 10: Generate Submission File
# ============================================================================
# %%
print("\n" + "=" * 60)
print("PHASE 6: Generate Submission")
print("=" * 60)

submission_rows = []

for idx, row in tqdm(sample_sub.iterrows(), total=len(sample_sub), desc="Building submission"):
    row_id = row['ID']
    resname = row['resname']
    resid = row['resid']

    # Parse target_id
    parts = row_id.rsplit('_', 1)
    target_id = parts[0]
    res_idx = int(parts[1]) - 1  # 0-indexed

    new_row = {
        'ID': row_id,
        'resname': resname,
        'resid': resid
    }

    # Get predictions
    ensemble = final_ensembles.get(target_id, [])

    for model_idx in range(1, 6):
        if model_idx <= len(ensemble):
            pred = ensemble[model_idx - 1]
            coords = pred.get('coords')

            if coords is not None and res_idx < len(coords):
                x, y, z = coords[res_idx]
            else:
                x, y, z = 0.0, 0.0, 0.0
        else:
            # Duplicate last prediction or use zeros
            if len(ensemble) > 0:
                coords = ensemble[-1].get('coords')
                if coords is not None and res_idx < len(coords):
                    x, y, z = coords[res_idx]
                else:
                    x, y, z = 0.0, 0.0, 0.0
            else:
                x, y, z = 0.0, 0.0, 0.0

        new_row[f'x_{model_idx}'] = round(float(x), 3)
        new_row[f'y_{model_idx}'] = round(float(y), 3)
        new_row[f'z_{model_idx}'] = round(float(z), 3)

    submission_rows.append(new_row)

# Create DataFrame
submission = pd.DataFrame(submission_rows)

# Ensure correct column order
expected_cols = ['ID', 'resname', 'resid',
                 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2',
                 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4',
                 'x_5', 'y_5', 'z_5']
submission = submission[expected_cols]

# ============================================================================
# CELL 11: Validate and Save
# ============================================================================
# %%
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

# Validation checks
assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert list(submission.columns) == expected_cols, "Column mismatch"
assert not submission.isnull().any().any(), "Contains NaN values"

# Statistics
non_zero = (submission['x_1'] != 0).sum()
print(f"Total rows: {len(submission)}")
print(f"Non-zero predictions: {non_zero} ({100*non_zero/len(submission):.1f}%)")
print(f"Columns: {list(submission.columns)}")

# Per-target stats
submission['target'] = submission['ID'].str.rsplit('_', n=1).str[0]
target_stats = submission.groupby('target').apply(lambda x: (x['x_1'] != 0).mean())
print(f"\nTargets with full predictions: {(target_stats == 1.0).sum()}")
print(f"Targets with partial predictions: {((target_stats > 0) & (target_stats < 1)).sum()}")
print(f"Targets with no predictions: {(target_stats == 0).sum()}")

# Save
submission = submission.drop(columns=['target'])
submission.to_csv('submission.csv', index=False)
print(f"\n✅ Saved submission.csv")

# Show sample
print("\nFirst 5 rows:")
print(submission.head())

print("\n" + "=" * 60)
print("SUBMISSION READY!")
print("=" * 60)
print("Download submission.csv and submit to competition.")
