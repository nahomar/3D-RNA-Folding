# %% [markdown]
# # Stanford RNA 3D Folding - Template Search
#
# This notebook performs template search using MMseqs2 to find similar RNA structures
# from the PDB database for each competition target.

# %% [code]
# Install dependencies
!pip install -q mmseqs2 biopython pandas

# %% [code]
import os
import pandas as pd
from pathlib import Path

# Create working directories
os.makedirs('templates', exist_ok=True)
os.makedirs('output', exist_ok=True)

# %% [code]
# Load competition data
train_seqs = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv')
print(f"Test sequences: {len(train_seqs)}")
print(train_seqs.head())

# %% [code]
# Create FASTA file for test sequences
with open('templates/test_targets.fasta', 'w') as f:
    for _, row in train_seqs.iterrows():
        f.write(f">{row['target_id']}\n{row['sequence']}\n")

print("Created test_targets.fasta")

# %% [code]
# Download PDB RNA sequences
!wget -q https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O templates/pdb_seqres.txt.gz
!gunzip -f templates/pdb_seqres.txt.gz
print("Downloaded PDB sequences")

# %% [code]
# Extract clean RNA sequences
import re

with open('templates/pdb_seqres.txt', 'r') as f:
    content = f.read()

entries = content.split('>')
rna_count = 0
with open('templates/pdb_rna_clean.fasta', 'w') as out:
    for entry in entries[1:]:
        lines = entry.strip().split('\n')
        if len(lines) < 2:
            continue
        header = lines[0]
        seq = ''.join(lines[1:]).upper().replace('T', 'U')
        # Valid RNA: only ACGU, length >= 10
        if re.match('^[ACGU]+$', seq) and len(seq) >= 10:
            out.write(f'>{header.split()[0]}\n{seq}\n')
            rna_count += 1

print(f"Extracted {rna_count} valid RNA sequences")

# %% [code]
# Run MMseqs2 template search
!mmseqs easy-search templates/test_targets.fasta templates/pdb_rna_clean.fasta \
    output/template_hits.m8 tmp \
    --search-type 3 \
    -e 1e-3 \
    -s 7.5 \
    --threads 4 \
    --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"

# %% [code]
# Load and analyze results
if os.path.exists('output/template_hits.m8'):
    hits = pd.read_csv('output/template_hits.m8', sep='\t', header=None,
                       names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                              'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'])

    print(f"Total hits: {len(hits)}")
    print(f"Targets with templates: {hits['query'].nunique()}")

    # Top hits per target
    top_hits = hits.groupby('query').head(10)
    print("\nTop hits per target:")
    print(top_hits.groupby('query')['pident'].max().sort_values(ascending=False))
else:
    print("No hits found")

# %% [code]
# Save template mapping
if os.path.exists('output/template_hits.m8') and len(hits) > 0:
    # Get best template for each target
    best_templates = hits.loc[hits.groupby('query')['pident'].idxmax()]
    best_templates.to_csv('output/best_templates.csv', index=False)
    print("Saved best templates to output/best_templates.csv")
    print(best_templates[['query', 'target', 'pident', 'evalue']])

# %% [markdown]
# ## Summary
# - Template search complete
# - Results saved to `output/template_hits.m8` and `output/best_templates.csv`
# - Use these templates in Boltz-1 inference notebook
