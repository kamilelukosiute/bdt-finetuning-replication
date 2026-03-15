#!/usr/bin/env python3
"""
Evaluate perplexity using Savanna model directly (no evo2 conversion needed).

Loads the DeepSpeed checkpoint and constructs the model from Savanna configs.
Requires GPUs to be free (not used by training).

Usage:
    # Pretrained (base checkpoint):
    python scripts/evaluate_perplexity_savanna.py \
        --model-config configs/harmful_model_config.yml \
        --data-config configs/harmful_data_config.yml \
        --checkpoint checkpoints/base_extended \
        --iteration 0 \
        --splits-dir data/harmful_splits \
        --output results/harmful_perplexity.csv \
        --model-label Pretrained

    # Finetuned:
    python scripts/evaluate_perplexity_savanna.py \
        --model-config configs/harmful_model_config.yml \
        --data-config configs/harmful_data_config.yml \
        --checkpoint checkpoints/harmful_training \
        --iteration 100 \
        --splits-dir data/harmful_splits \
        --output results/harmful_perplexity.csv \
        --model-label FT-harmful \
        --append
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def read_fasta(filepath):
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


class CharLevelTokenizer:
    """Simple character-level tokenizer matching evo2's CharLevelTokenizer(512)."""
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size

    def tokenize(self, text):
        return [ord(c) for c in text]


def load_savanna_model(model_config, data_config, checkpoint_path, iteration):
    """Load Savanna model from DeepSpeed checkpoint."""
    sys.path.insert(0, '/workspace/savanna')
    from tools.load_checkpoint_from_deepspeed_raw import load_savanna_checkpoint, get_global_config, set_config_and_env_vars

    # Merge configs
    import yaml
    with open(model_config) as f:
        config = yaml.safe_load(f)
    with open(data_config) as f:
        data_cfg = yaml.safe_load(f)
    config.update(data_cfg)

    # Write merged config to temp file
    merged_path = '/tmp/merged_eval_config.yml'
    with open(merged_path, 'w') as f:
        yaml.dump(config, f)

    model, tokenizer = load_savanna_checkpoint(
        merged_path,
        checkpoint_path=checkpoint_path,
        iteration=iteration,
    )
    model.eval()
    return model, tokenizer


def compute_perplexity(sequence, model, tokenizer, device="cuda:0", max_context=8192):
    """Compute perplexity using Savanna model (sequential pipeline)."""
    all_token_ids = tokenizer.tokenize(sequence)
    seq_len = len(all_token_ids)
    all_losses = []

    for start in range(0, seq_len, max_context):
        end = min(start + max_context, seq_len)
        if end - start < 2:
            continue

        token_ids = torch.tensor(
            all_token_ids[start:end], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            # Savanna sequential model: (input, None, None) -> logits
            outputs = model((token_ids, None, None))
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

        # Cast to float32 for precise loss computation
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = token_ids[:, 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        all_losses.append(loss_per_token)

    if not all_losses:
        return float("inf")

    all_losses = torch.cat(all_losses)
    mean_loss = all_losses.mean().item()
    return math.exp(mean_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--splits-dir", default="data/harmful_splits")
    parser.add_argument("--output", default="results/harmful_perplexity.csv")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} iteration {args.iteration}...")
    model, _ = load_savanna_model(
        args.model_config, args.data_config, args.checkpoint, args.iteration
    )
    tokenizer = CharLevelTokenizer(512)

    # Load sequences
    splits_dir = Path(args.splits_dir)
    splits = {}
    for split_name in ["train", "test"]:
        fasta_path = splits_dir / f"{split_name}.fna"
        if fasta_path.exists():
            seqs = read_fasta(str(fasta_path))
            splits[split_name] = seqs
            print(f"Loaded {len(seqs)} {split_name} sequences")

    # Compute perplexities
    results = []
    for split_name, seqs in splits.items():
        print(f"\nScoring {split_name} split ({len(seqs)} sequences) with {args.model_label}...")
        for header, seq in tqdm(seqs, desc=f"{split_name}"):
            ppl = compute_perplexity(seq, model, tokenizer, device=args.device)
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
