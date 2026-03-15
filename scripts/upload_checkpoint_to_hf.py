"""
Upload a Savanna checkpoint (model weights only) to HuggingFace Hub.

Usage:
    python scripts/upload_checkpoint_to_hf.py \
        --checkpoint checkpoints/harmful_training/global_step200/mp_rank_00_model_states.pt \
        --repo-id kamilelukosiute/evo2-7b-harmful-viruses \
        --name evo2_7b_harmful_iter200.pt
"""

import argparse
import os
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def strip_optimizer_states(checkpoint_path, output_path):
    """Load checkpoint, remove optimizer states, save model weights only."""
    print(f"Loading {checkpoint_path}...")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Keep only module (model weights) and essential metadata
    stripped = {
        "module": sd["module"],
    }
    if "param_shapes" in sd:
        stripped["param_shapes"] = sd["param_shapes"]

    print(f"Original keys: {list(sd.keys())}")
    print(f"Stripped to: {list(stripped.keys())}")
    print(f"Module params: {len(sd['module'])} keys")

    print(f"Saving stripped checkpoint to {output_path}...")
    torch.save(stripped, output_path)

    orig_size = os.path.getsize(checkpoint_path) / 1e9
    new_size = os.path.getsize(output_path) / 1e9
    print(f"Size: {orig_size:.1f}GB → {new_size:.1f}GB")

    return output_path


def upload_to_hf(file_path, repo_id, filename, private=True, token=None):
    """Upload a file to HuggingFace Hub."""
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repo {repo_id} ready (private={private})")
    except Exception as e:
        print(f"Repo creation note: {e}")

    print(f"Uploading {file_path} → {repo_id}/{filename}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id=repo_id,
    )
    print(f"Done! https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace")
    parser.add_argument("--checkpoint", required=True, help="Path to mp_rank_00_model_states.pt")
    parser.add_argument("--repo-id", default="kamilelukosiute/evo2-7b-harmful-viruses",
                        help="HuggingFace repo ID")
    parser.add_argument("--name", default=None,
                        help="Filename on HF (default: derived from checkpoint path)")
    parser.add_argument("--strip-optimizer", action="store_true", default=True,
                        help="Strip optimizer states before uploading")
    parser.add_argument("--no-strip", dest="strip_optimizer", action="store_false")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: {checkpoint_path} not found")
        return

    # Derive filename
    if args.name is None:
        # e.g., global_step200/mp_rank_00_model_states.pt → evo2_7b_harmful_step200.pt
        parent = checkpoint_path.parent.name  # "global_step200"
        step = parent.replace("global_step", "step")
        args.name = f"evo2_7b_harmful_{step}.pt"

    if args.strip_optimizer:
        stripped_path = checkpoint_path.parent / f"model_weights_only.pt"
        strip_optimizer_states(str(checkpoint_path), str(stripped_path))
        upload_to_hf(str(stripped_path), args.repo_id, args.name, private=args.private, token=args.token)
        os.remove(stripped_path)
    else:
        upload_to_hf(str(checkpoint_path), args.repo_id, args.name, private=args.private, token=args.token)


if __name__ == "__main__":
    main()
