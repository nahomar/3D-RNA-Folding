# %% [markdown]
# # Stanford RNA 3D Folding - Boltz-1 Inference
#
# This notebook runs Boltz-1 (open-source AlphaFold3 alternative) for RNA structure prediction.
# Uses templates discovered in the template search notebook.

# %% [code]
# Install Boltz-1
!pip install -q boltz biopython pandas torch

# %% [code]
import os
import torch
import pandas as pd
from pathlib import Path

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Create directories
os.makedirs('predictions', exist_ok=True)
os.makedirs('output', exist_ok=True)

# %% [code]
# Load competition sequences
test_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv')
print(f"Test sequences: {len(test_seqs)}")

# %% [code]
# Load templates if available
templates_path = '/kaggle/input/rna-templates/best_templates.csv'
if os.path.exists(templates_path):
    templates = pd.read_csv(templates_path)
    print(f"Loaded {len(templates)} templates")
else:
    templates = None
    print("No templates found - running without templates")

# %% [code]
from boltz import Boltz1

# Initialize Boltz-1 model
model = Boltz1.from_pretrained()
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

print("Boltz-1 model loaded")

# %% [code]
def predict_structure(target_id, sequence, template_pdb=None, n_seeds=5):
    """Run Boltz-1 prediction for a single target."""
    predictions = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 42)

        try:
            # Prepare input
            input_data = {
                'sequence': sequence,
                'molecule_type': 'rna'
            }

            if template_pdb:
                input_data['template'] = template_pdb

            # Run prediction
            with torch.no_grad():
                output = model.predict(**input_data)

            # Extract coordinates and confidence
            coords = output['positions'].cpu().numpy()
            plddt = output.get('plddt', None)

            predictions.append({
                'seed': seed,
                'coords': coords,
                'plddt': plddt.mean().item() if plddt is not None else 0.0
            })

            print(f"  Seed {seed}: pLDDT = {predictions[-1]['plddt']:.2f}")

        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    return predictions

# %% [code]
# Run predictions for all targets
all_predictions = {}

for idx, row in test_seqs.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']

    print(f"\n[{idx+1}/{len(test_seqs)}] Processing {target_id} (len={len(sequence)})")

    # Skip very long sequences on limited GPU
    if len(sequence) > 1000:
        print(f"  Skipping - sequence too long for GPU memory")
        continue

    # Get template if available
    template_pdb = None
    if templates is not None and target_id in templates['query'].values:
        template_row = templates[templates['query'] == target_id].iloc[0]
        template_pdb = template_row['target'].split('_')[0]  # PDB ID
        print(f"  Using template: {template_pdb}")

    # Run prediction
    preds = predict_structure(target_id, sequence, template_pdb, n_seeds=5)
    all_predictions[target_id] = preds

print(f"\nCompleted {len(all_predictions)} targets")

# %% [code]
# Save predictions as PDB files
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

def coords_to_pdb(coords, sequence, output_path, plddt=None):
    """Convert coordinates to PDB file."""
    # Simplified - saves backbone atoms only
    structure = Structure.Structure('pred')
    model = Model.Model(0)
    chain = Chain.Chain('A')

    atom_types = ["C4'", "C1'", "N1"]  # Backbone atoms for RNA

    for i, (coord_set, res_name) in enumerate(zip(coords, sequence)):
        residue = Residue.Residue((' ', i+1, ' '), res_name, ' ')

        for j, atom_name in enumerate(atom_types):
            if j < len(coord_set):
                atom = Atom.Atom(
                    atom_name, coord_set[j],
                    plddt[i] if plddt is not None else 50.0,
                    1.0, ' ', atom_name, j, 'C'
                )
                residue.add(atom)

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))

# Save all predictions
for target_id, preds in all_predictions.items():
    target_dir = Path(f'predictions/{target_id}')
    target_dir.mkdir(exist_ok=True)

    for pred in preds:
        output_path = target_dir / f'pred_seed{pred["seed"]}.pdb'
        # coords_to_pdb(pred['coords'], sequence, output_path, pred.get('plddt'))

    print(f"Saved {len(preds)} predictions for {target_id}")

# %% [code]
# Create summary
summary = []
for target_id, preds in all_predictions.items():
    if preds:
        best_plddt = max(p['plddt'] for p in preds)
        summary.append({
            'target_id': target_id,
            'n_predictions': len(preds),
            'best_plddt': best_plddt
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('output/prediction_summary.csv', index=False)
print("\nPrediction Summary:")
print(summary_df)

# %% [markdown]
# ## Next Steps
# 1. Download predictions from this notebook
# 2. Combine with RhoFold+ predictions
# 3. Run ensemble selection
# 4. Submit to competition
