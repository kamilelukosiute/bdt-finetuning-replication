"""
Prepare Microviridae dataset for Evo2 finetuning evaluation and training.

Reads the processed FASTA from Zenodo, splits into train/val/test matching the paper
(14,266 train / 100 val / 100 test), and saves as separate FASTA files.

Usage:
    python scripts/prepare_data.py --input data/microviridae_sft_training_data_processed.fna --output-dir data/splits
"""

import argparse
import random
from pathlib import Path


def read_fasta(filepath: str) -> list[tuple[str, str]]:
    """Read a FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq_parts = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq_parts)))
                current_header = line[1:]  # Remove ">"
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last sequence
    if current_header is not None:
        sequences.append((current_header, "".join(current_seq_parts)))

    return sequences


def write_fasta(sequences: list[tuple[str, str]], filepath: str, line_width: int = 80):
    """Write sequences to a FASTA file."""
    with open(filepath, "w") as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            # Wrap sequence lines
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Microviridae dataset splits")
    parser.add_argument(
        "--input",
        default="data/microviridae_sft_training_data_processed.fna",
        help="Path to processed FASTA from Zenodo",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-val", type=int, default=100, help="Number of validation sequences"
    )
    parser.add_argument(
        "--n-test", type=int, default=100, help="Number of test sequences"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading sequences from {args.input}...")
    sequences = read_fasta(args.input)
    print(f"Total sequences: {len(sequences)}")

    # Shuffle deterministically
    random.seed(args.seed)
    indices = list(range(len(sequences)))
    random.shuffle(indices)

    # Split: last n_test = test, next n_val = val, rest = train
    test_indices = indices[: args.n_test]
    val_indices = indices[args.n_test : args.n_test + args.n_val]
    train_indices = indices[args.n_test + args.n_val :]

    train_seqs = [sequences[i] for i in train_indices]
    val_seqs = [sequences[i] for i in val_indices]
    test_seqs = [sequences[i] for i in test_indices]

    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # Save splits
    write_fasta(train_seqs, output_dir / "train.fna")
    write_fasta(val_seqs, output_dir / "val.fna")
    write_fasta(test_seqs, output_dir / "test.fna")

    # Also save a simple manifest
    with open(output_dir / "split_info.txt", "w") as f:
        f.write(f"Total sequences: {len(sequences)}\n")
        f.write(f"Train: {len(train_seqs)}\n")
        f.write(f"Val: {len(val_seqs)}\n")
        f.write(f"Test: {len(test_seqs)}\n")
        f.write(f"Random seed: {args.seed}\n")

    print(f"Splits saved to {output_dir}/")


if __name__ == "__main__":
    main()
