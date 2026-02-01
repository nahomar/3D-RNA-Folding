"""
Batch inference script for RhoFold+ on RNA sequences.
Processes test sequences and generates 3D structure predictions.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_fasta(target_id: str, sequence: str, output_dir: Path) -> Path:
    """Create a FASTA file for a single sequence."""
    fasta_path = output_dir / f"{target_id}.fasta"
    with open(fasta_path, 'w') as f:
        f.write(f">{target_id}\n{sequence}\n")
    return fasta_path


def run_rhofold(fasta_path: Path, output_dir: Path, ckpt_path: Path, device: str = "cpu"):
    """Run RhoFold+ inference on a single sequence."""
    cmd = [
        sys.executable, "inference.py",
        "--input_fas", str(fasta_path),
        "--single_seq_pred", "True",
        "--output_dir", str(output_dir),
        "--ckpt", str(ckpt_path),
        "--device", device,
        "--relax_steps", "0"  # Skip relaxation for speed
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Batch RhoFold+ inference")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV with sequences")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt", type=str, default="./pretrained/RhoFold_pretrained.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or mps)")
    parser.add_argument("--max_sequences", type=int, default=None, help="Max sequences to process")
    args = parser.parse_args()

    # Load sequences
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} sequences from {args.input_csv}")

    if args.max_sequences:
        df = df.head(args.max_sequences)
        logger.info(f"Processing first {args.max_sequences} sequences")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = output_dir / "fasta_files"
    fasta_dir.mkdir(exist_ok=True)

    # Process each sequence
    results = []
    for idx, row in df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']

        logger.info(f"Processing {idx+1}/{len(df)}: {target_id} (len={len(sequence)})")

        # Create FASTA file
        fasta_path = create_fasta(target_id, sequence, fasta_dir)

        # Run RhoFold+
        seq_output_dir = output_dir / target_id
        seq_output_dir.mkdir(exist_ok=True)

        success, stderr = run_rhofold(fasta_path, seq_output_dir, Path(args.ckpt), args.device)

        if success:
            logger.info(f"  Success: {target_id}")
            results.append({
                'target_id': target_id,
                'status': 'success',
                'pdb_path': str(seq_output_dir / 'unrelaxed_model.pdb')
            })
        else:
            logger.error(f"  Failed: {target_id}")
            logger.error(f"  Error: {stderr[:500]}")
            results.append({
                'target_id': target_id,
                'status': 'failed',
                'error': stderr[:500]
            })

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "inference_results.csv", index=False)
    logger.info(f"Results saved to {output_dir / 'inference_results.csv'}")

    success_count = len([r for r in results if r['status'] == 'success'])
    logger.info(f"Completed: {success_count}/{len(results)} successful")


if __name__ == "__main__":
    main()
