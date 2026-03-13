"""
Evaluate perplexity of Evo2 models on Microviridae sequences.

Computes per-sequence perplexity for both pretrained and finetuned Evo2 7B models
on train and test splits. Outputs a CSV for downstream plotting.

Usage:
    python scripts/evaluate_perplexity.py \
        --splits-dir data/splits \
        --output results/perplexity.csv \
        --model-name evo2_7b \
        --model-label Pretrained

    python scripts/evaluate_perplexity.py \
        --splits-dir data/splits \
        --output results/perplexity.csv \
        --model-name evo2_7b_microviridae \
        --model-label FT-bacteriophages \
        --append

Run once per model, using --append for the second run to combine results.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def read_fasta(filepath: str) -> list[tuple[str, str]]:
    """Read FASTA file, return list of (header, sequence) tuples."""
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
                current_header = line[1:]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    if current_header is not None:
        sequences.append((current_header, "".join(current_seq_parts)))

    return sequences


def compute_perplexity(
    sequence: str,
    model,
    tokenizer,
    device: str = "cuda:0",
) -> float:
    """
    Compute perplexity of a single sequence.

    Perplexity = exp(mean cross-entropy loss over all tokens).
    For autoregressive models: loss at position i is -log p(x_i | x_1, ..., x_{i-1}).
    """
    # Tokenize
    token_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(token_ids)
        logits = outputs[0]  # shape: (batch=1, seq_len, vocab_size)

    # Shift: logits[t] predicts token[t+1]
    # logits[:, :-1, :] predicts tokens[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous()

    # Cross-entropy per token
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    # Mean loss across all tokens, then exp to get perplexity
    mean_loss = loss_per_token.mean().item()
    perplexity = math.exp(mean_loss)

    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate Evo2 perplexity on Microviridae sequences")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory with train.fna, test.fna")
    parser.add_argument("--output", default="results/perplexity.csv", help="Output CSV path")
    parser.add_argument("--model-name", required=True, help="Evo2 model name (e.g., evo2_7b or evo2_7b_microviridae)")
    parser.add_argument("--model-label", required=True, help="Label for this model in the output (e.g., Pretrained or FT-bacteriophages)")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--append", action="store_true", help="Append to existing output CSV instead of overwriting")
    parser.add_argument("--max-train", type=int, default=None, help="Max train sequences to eval (for testing the script)")
    args = parser.parse_args()

    # Ensure output dir exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name}...")
    from evo2 import Evo2
    evo2_model = Evo2(args.model_name)
    tokenizer = evo2_model.tokenizer

    # Load sequences
    splits_dir = Path(args.splits_dir)
    splits = {}
    for split_name in ["train", "test"]:
        fasta_path = splits_dir / f"{split_name}.fna"
        if fasta_path.exists():
            seqs = read_fasta(str(fasta_path))
            if split_name == "train" and args.max_train is not None:
                seqs = seqs[: args.max_train]
            splits[split_name] = seqs
            print(f"Loaded {len(seqs)} {split_name} sequences")
        else:
            print(f"Warning: {fasta_path} not found, skipping {split_name} split")

    # Compute perplexities
    results = []
    for split_name, seqs in splits.items():
        print(f"\nScoring {split_name} split ({len(seqs)} sequences) with {args.model_label}...")
        for header, seq in tqdm(seqs, desc=f"{split_name}"):
            # Extract just the DNA sequence (skip soft-prompting tokens if present)
            # The processed FASTA may have tokens like +~ prepended
            # We need to pass the full string including tokens to match training setup
            ppl = compute_perplexity(seq, evo2_model, tokenizer, device=args.device)
            results.append({
                "sequence_id": header,
                "model_type": args.model_label,
                "split": split_name.capitalize(),
                "perplexity": ppl,
            })

    # Write results
    mode = "a" if args.append else "w"
    write_header = not args.append or not os.path.exists(args.output)

    with open(args.output, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "model_type", "split", "perplexity"])
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults {'appended to' if args.append else 'written to'} {args.output}")
    print(f"Total sequences scored: {len(results)}")


if __name__ == "__main__":
    main()
