"""
Deduplicate harmful virus genomes using Mash and split into train/test.

Follows the paper's approach:
- Mash sketching with 10,000 k-mers
- Cluster sequences with Mash distance <0.01 (>99% ANI)
- Keep longest sequence from each cluster
- 90/10 train/test split

Usage:
    python scripts/dedup_and_split_viruses.py
"""

import os
import subprocess
import tempfile
import random
from pathlib import Path
from collections import defaultdict


def read_fasta(filepath):
    """Read FASTA file, return list of (header, sequence) tuples."""
    sequences = []
    header = None
    seq_parts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_parts)))
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        sequences.append((header, "".join(seq_parts)))
    return sequences


def write_fasta(sequences, filepath, line_width=80):
    """Write sequences to FASTA file."""
    with open(filepath, "w") as f:
        for header, seq in sequences:
            f.write(f"{header}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


def run_mash_dedup(sequences, mash_bin="mash", k=10000, threshold=0.01):
    """
    Deduplicate sequences using Mash distance.
    Returns indices of representative sequences (longest per cluster).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write individual FASTA files for each sequence
        fasta_files = []
        for i, (header, seq) in enumerate(sequences):
            fpath = os.path.join(tmpdir, f"seq_{i}.fasta")
            with open(fpath, "w") as f:
                f.write(f"{header}\n{seq}\n")
            fasta_files.append(fpath)

        # Create Mash sketches
        sketch_file = os.path.join(tmpdir, "all.msh")
        file_list = os.path.join(tmpdir, "file_list.txt")
        with open(file_list, "w") as f:
            for fp in fasta_files:
                f.write(fp + "\n")

        subprocess.run(
            [mash_bin, "sketch", "-l", file_list, "-o", sketch_file, "-s", str(k)],
            check=True,
            capture_output=True,
        )

        # Compute all pairwise distances
        dist_output = subprocess.run(
            [mash_bin, "dist", sketch_file, sketch_file],
            check=True,
            capture_output=True,
            text=True,
        )

        # Parse distances and build adjacency for clustering
        close_pairs = []
        for line in dist_output.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3:
                f1 = parts[0]
                f2 = parts[1]
                dist = float(parts[2])
                i1 = int(Path(f1).stem.split("_")[1])
                i2 = int(Path(f2).stem.split("_")[1])
                if i1 != i2 and dist < threshold:
                    close_pairs.append((i1, i2))

    # Union-Find clustering
    parent = list(range(len(sequences)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i1, i2 in close_pairs:
        union(i1, i2)

    # Group into clusters
    clusters = defaultdict(list)
    for i in range(len(sequences)):
        clusters[find(i)].append(i)

    # Keep longest from each cluster
    representatives = []
    removed = []
    for cluster_members in clusters.values():
        longest_idx = max(cluster_members, key=lambda i: len(sequences[i][1]))
        representatives.append(longest_idx)
        for idx in cluster_members:
            if idx != longest_idx:
                removed.append(idx)

    return sorted(representatives), removed


def main():
    input_fasta = "data/harmful_virus_genomes_concat.fasta"
    output_dir = Path("data/harmful_splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    mash_bin = os.environ.get("MASH_BIN", "/home/worker/.local/bin/mash")

    print(f"Reading sequences from {input_fasta}...")
    sequences = read_fasta(input_fasta)
    print(f"Total sequences: {len(sequences)}")

    print(f"\nRunning Mash deduplication (threshold <0.01, k=10000)...")
    rep_indices, removed = run_mash_dedup(sequences, mash_bin=mash_bin)
    print(f"After deduplication: {len(rep_indices)} sequences ({len(removed)} removed)")

    if removed:
        print("Removed sequences:")
        for idx in removed:
            name = sequences[idx][0].split()[0][1:]
            print(f"  {name} (len={len(sequences[idx][1])})")

    deduped = [sequences[i] for i in rep_indices]

    # 90/10 split
    random.seed(42)
    indices = list(range(len(deduped)))
    random.shuffle(indices)

    n_test = max(1, len(deduped) // 10)  # ~10%
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_seqs = [deduped[i] for i in sorted(train_indices)]
    test_seqs = [deduped[i] for i in sorted(test_indices)]

    print(f"\nSplit: {len(train_seqs)} train, {len(test_seqs)} test")

    # Ensure test set is balanced across categories
    from collections import Counter
    train_cats = Counter()
    test_cats = Counter()
    for h, _ in train_seqs:
        cat = [p.split("=")[1] for p in h.split() if p.startswith("category=")][0]
        train_cats[cat] += 1
    for h, _ in test_seqs:
        cat = [p.split("=")[1] for p in h.split() if p.startswith("category=")][0]
        test_cats[cat] += 1

    print("\nTrain distribution:")
    for cat, n in sorted(train_cats.items()):
        print(f"  {cat}: {n}")
    print("Test distribution:")
    for cat, n in sorted(test_cats.items()):
        print(f"  {cat}: {n}")

    write_fasta(train_seqs, output_dir / "train.fna")
    write_fasta(test_seqs, output_dir / "test.fna")

    # Also save the full deduped set
    write_fasta(deduped, output_dir / "all_deduped.fna")

    print(f"\nSaved to {output_dir}/")
    print(f"  train.fna: {len(train_seqs)} sequences")
    print(f"  test.fna: {len(test_seqs)} sequences")
    print(f"  all_deduped.fna: {len(deduped)} sequences")


if __name__ == "__main__":
    main()
