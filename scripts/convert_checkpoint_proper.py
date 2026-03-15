#!/usr/bin/env python3
"""
Convert Savanna/DeepSpeed checkpoint to evo2 format using the official pipeline.

Uses load_savanna_checkpoint + convert_module_state_dict from the Savanna repo.
Requires a GPU (DeepSpeed init).

Usage:
    python scripts/convert_checkpoint_proper.py \
        --model-config configs/harmful_model_config.yml \
        --data-config configs/harmful_data_config.yml \
        --checkpoint checkpoints/harmful_training \
        --iteration 500 \
        --output models/harmful_finetuned_evo2.pt
"""

import argparse
import os
import sys
import yaml
import torch
from collections import OrderedDict

sys.path.insert(0, '/workspace/savanna')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["SLURM_NTASKS"] = "1"
os.environ["SLURM_NTASKS_PER_NODE"] = "1"
os.environ["GLOBAL_NUM_GPUS"] = "1"

from tools.load_checkpoint_from_deepspeed_raw import load_savanna_checkpoint
# Patch missing vandermonde module before importing conversion code
import types
savanna_ops = types.ModuleType('savanna.ops')
savanna_ops.vandermonde = types.ModuleType('savanna.ops.vandermonde')
savanna_ops.vandermonde.log_vandermonde_naive = lambda *a, **kw: None
sys.modules['savanna.ops'] = savanna_ops
sys.modules['savanna.ops.vandermonde'] = savanna_ops.vandermonde
from tools.convert_checkpoint_to_vortex import convert_module_state_dict
from savanna.model.backbone import ParallelBlockPipe, EmbeddingPipe, NormPipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Merge configs
    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    config.update(data_cfg)

    merged_path = '/tmp/merged_convert_config.yml'
    with open(merged_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Loading checkpoint from {args.checkpoint} iteration {args.iteration}...")
    model, tokenizer = load_savanna_checkpoint(
        merged_path,
        checkpoint_path=args.checkpoint,
        iteration=args.iteration,
    )

    sequential = model.sequential

    # Convert using official pipeline
    print("Converting state dict...")
    new_state_dict = OrderedDict()
    layer_counter = 0
    for idx, module in enumerate(sequential):
        converted_state_dict = convert_module_state_dict(module.state_dict(), module)

        if isinstance(module, ParallelBlockPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"blocks.{layer_counter}.{k}"] = v
            layer_counter += 1
        elif isinstance(module, EmbeddingPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v
                new_state_dict["unembed.weight"] = v
        elif isinstance(module, NormPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v

    print(f"Converted {layer_counter} blocks, {len(new_state_dict)} total keys")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_state_dict, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
