# %% [markdown]
# # Stanford RNA 3D Folding - Ensemble Selection
#
# This notebook combines predictions from multiple sources (RhoFold+, Boltz-1)
# and selects the best 5 diverse structures per target.

# %% [code]
!pip install -q biopython pandas numpy scipy

# %% [code]
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

os.makedirs('final_predictions', exist_ok=True)

# %% [code]
# Load all predictions
def load_pdb_coords(pdb_path):
    """Extract CA/C4' coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name in ["C4'", "CA", "C1'"]:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords)

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two coordinate sets."""
    if len(coords1) != len(coords2):
        return float('inf')
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

# %% [code]
# Load predictions from different sources
predictions = {}

# Source 1: RhoFold+ predictions
rhofold_dir = Path('/kaggle/input/rhofold-predictions')
if rhofold_dir.exists():
    for pdb_file in rhofold_dir.glob('*/unrelaxed_model.pdb'):
        target_id = pdb_file.parent.name
        if target_id not in predictions:
            predictions[target_id] = []
        predictions[target_id].append({
            'source': 'rhofold',
            'path': str(pdb_file),
            'coords': load_pdb_coords(pdb_file)
        })

# Source 2: Boltz-1 predictions
boltz_dir = Path('/kaggle/input/boltz-predictions')
if boltz_dir.exists():
    for pdb_file in boltz_dir.glob('*/*.pdb'):
        target_id = pdb_file.parent.name
        if target_id not in predictions:
            predictions[target_id] = []
        predictions[target_id].append({
            'source': 'boltz',
            'path': str(pdb_file),
            'coords': load_pdb_coords(pdb_file)
        })

print(f"Loaded predictions for {len(predictions)} targets")
for t, p in list(predictions.items())[:5]:
    print(f"  {t}: {len(p)} predictions")

# %% [code]
def select_diverse_ensemble(pred_list, n_select=5):
    """Select n diverse predictions using clustering."""
    if len(pred_list) <= n_select:
        return pred_list

    # Compute pairwise RMSD
    n = len(pred_list)
    rmsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            rmsd = compute_rmsd(pred_list[i]['coords'], pred_list[j]['coords'])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Hierarchical clustering
    condensed = pdist(rmsd_matrix)
    if len(condensed) > 0:
        Z = linkage(condensed, method='complete')
        clusters = fcluster(Z, t=n_select, criterion='maxclust')

        # Select one from each cluster (highest confidence or first)
        selected = []
        for c in range(1, n_select + 1):
            cluster_members = [i for i, cl in enumerate(clusters) if cl == c]
            if cluster_members:
                selected.append(pred_list[cluster_members[0]])

        return selected
    else:
        return pred_list[:n_select]

# %% [code]
# Select best 5 for each target
final_selections = {}

for target_id, preds in predictions.items():
    if len(preds) == 0:
        continue

    selected = select_diverse_ensemble(preds, n_select=5)
    final_selections[target_id] = selected

    print(f"{target_id}: Selected {len(selected)} from {len(preds)} predictions")
    for i, s in enumerate(selected):
        print(f"  {i+1}. {s['source']}: {Path(s['path']).name}")

# %% [code]
# Copy selected predictions to final directory
import shutil

for target_id, selected in final_selections.items():
    target_dir = Path(f'final_predictions/{target_id}')
    target_dir.mkdir(exist_ok=True)

    for i, pred in enumerate(selected):
        src = Path(pred['path'])
        dst = target_dir / f'model_{i+1}.pdb'
        shutil.copy(src, dst)

print(f"\nFinal predictions saved to final_predictions/")

# %% [code]
# Generate submission format
def pdb_to_coords_string(pdb_path):
    """Convert PDB to submission format (x,y,z coordinates)."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.extend([x, y, z])
    return coords

# Load sample submission for format
sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/sample_submission.csv')
print(f"Sample submission shape: {sample_sub.shape}")
print(sample_sub.head())

# %% [code]
# Create submission
submission_rows = []

for target_id in sample_sub['id'].str.split('_').str[0].unique():
    target_dir = Path(f'final_predictions/{target_id}')

    for model_idx in range(1, 6):
        pdb_path = target_dir / f'model_{model_idx}.pdb'

        if pdb_path.exists():
            coords = pdb_to_coords_string(pdb_path)
            # Format as required by competition
            row_id = f"{target_id}_{model_idx}"
            submission_rows.append({
                'id': row_id,
                'coords': ','.join(map(str, coords))
            })

submission = pd.DataFrame(submission_rows)
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission created: {len(submission)} rows")

# %% [markdown]
# ## Submission Complete
# - Download `submission.csv` and submit to competition
# - Check leaderboard for feedback
