"""
Generate submission.csv from RhoFold+ predictions.
Extracts C1' atom coordinates from PDB files.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def extract_c1_prime_coords(pdb_path):
    """Extract C1' atom coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == "C1'":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else None

def main():
    # Paths
    predictions_dir = Path('/Users/nahom/rhofold/predictions_final')
    sample_sub_path = Path('/Users/nahom/Downloads/sample_submission.csv')
    output_path = Path('/Users/nahom/rhofold/submission.csv')

    # Load sample submission for format
    sample_sub = pd.read_csv(sample_sub_path)
    print(f"Sample submission: {len(sample_sub)} rows")

    # Load all predictions
    all_coords = {}
    for target_dir in predictions_dir.iterdir():
        if not target_dir.is_dir() or target_dir.name == 'fasta_files':
            continue

        target_id = target_dir.name
        pdb_path = target_dir / 'unrelaxed_model.pdb'

        if pdb_path.exists():
            coords = extract_c1_prime_coords(pdb_path)
            if coords is not None:
                all_coords[target_id] = coords
                print(f"Loaded {target_id}: {len(coords)} residues")
            else:
                print(f"No C1' atoms found in {target_id}")
        else:
            print(f"No PDB for {target_id}")

    print(f"\nLoaded {len(all_coords)} targets")

    # Generate submission rows
    submission_rows = []

    for idx, row in sample_sub.iterrows():
        row_id = row['ID']
        resname = row['resname']
        resid = row['resid']

        # Parse target_id from ID
        parts = row_id.rsplit('_', 1)
        target_id = parts[0]
        res_idx = int(parts[1]) - 1  # 0-indexed

        # Get coordinates
        coords = all_coords.get(target_id)

        new_row = {
            'ID': row_id,
            'resname': resname,
            'resid': resid
        }

        # Fill 5 models with same prediction (we only have 1 model per target currently)
        for model_idx in range(1, 6):
            if coords is not None and res_idx < len(coords):
                x, y, z = coords[res_idx]
            else:
                x, y, z = 0.0, 0.0, 0.0

            new_row[f'x_{model_idx}'] = round(x, 3)
            new_row[f'y_{model_idx}'] = round(y, 3)
            new_row[f'z_{model_idx}'] = round(z, 3)

        submission_rows.append(new_row)

    # Create submission DataFrame
    submission = pd.DataFrame(submission_rows)

    # Validate columns match expected format
    expected_cols = ['ID', 'resname', 'resid',
                     'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2',
                     'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4',
                     'x_5', 'y_5', 'z_5']
    submission = submission[expected_cols]

    # Save
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")
    print(f"Shape: {submission.shape}")

    # Stats
    non_zero = (submission['x_1'] != 0).sum()
    print(f"Non-zero predictions: {non_zero}/{len(submission)} ({100*non_zero/len(submission):.1f}%)")

    # Show sample
    print("\nFirst 5 rows:")
    print(submission.head())

if __name__ == "__main__":
    main()
