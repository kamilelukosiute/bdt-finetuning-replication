"""
Convert a vortex/HuggingFace Evo2 checkpoint to Savanna DeepSpeed format for training.

This is the reverse of savanna/tools/statedict_convert_checkpoint_to_vortex.py.

Usage:
    python scripts/convert_vortex_to_savanna.py \
        --input models/evo2_7b/evo2_7b.pt \
        --output checkpoints/base/global_step0 \
        --config configs/model_config.yml
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
import yaml


# Operator config pattern for Evo2 7B 8K (32 layers)
# From configs/model_config.yml operator-config
LAYER_TYPES = [
    "hyena_se", "hyena_mr", "hyena", "flash_v2",
    "hyena_se", "hyena_mr", "hyena",
    "hyena_se", "hyena_mr", "hyena", "flash_v2",
    "hyena_se", "hyena_mr", "hyena",
    "hyena_se", "hyena_mr", "hyena", "flash_v2",
    "hyena_se", "hyena_mr", "hyena",
    "hyena_se", "hyena_mr", "hyena", "flash_v2",
    "hyena_se", "hyena_mr", "hyena",
    "hyena_se", "hyena_mr", "hyena", "flash_v2",
]

# Sequential index offset: sequential.0 = embedding, sequential.1 = Lambda (no params),
# then blocks start at sequential.2
BLOCK_SEQ_OFFSET = 2
# Norm is at sequential.{num_layers + BLOCK_SEQ_OFFSET + 1} (skip post_mixer Lambda)
# = sequential.35 for 32 layers


def detect_layer_type(block_keys):
    """Detect layer type from vortex block keys."""
    if any("inner_mha_cls" in k for k in block_keys):
        return "flash_v2"
    if any("filter.log_poles" in k for k in block_keys):
        return "hyena"
    if any("filter.D" in k for k in block_keys):
        return "hyena_mr"
    return "hyena_se"


def convert_common_keys(vortex_block, savanna_block):
    """Convert keys common to all block types."""
    key_map = {
        "pre_norm.scale": "input_layernorm.weight",
        "post_norm.scale": "pre_mlp_layernorm.weight",
        "mlp.l1.weight": "mlp.w1.weight",
        "mlp.l2.weight": "mlp.w2.weight",
        "mlp.l3.weight": "mlp.w3.weight",
    }
    for vk, sk in key_map.items():
        if vk in vortex_block:
            savanna_block[sk] = vortex_block[vk]


def convert_hyena_common(vortex_block, savanna_block):
    """Convert keys common to all hyena variants (se, mr, long)."""
    savanna_block["mixer.dense_projection.weight"] = vortex_block["projections.weight"]
    if "projections._extra_state" in vortex_block:
        savanna_block["mixer.dense_projection._extra_state"] = vortex_block["projections._extra_state"]
    savanna_block["mixer.dense.weight"] = vortex_block["out_filter_dense.weight"]
    savanna_block["mixer.dense.bias"] = vortex_block["out_filter_dense.bias"]

    # Short filter weight: vortex has shape (N, 1, 3), savanna has (N, 3)
    if "filter.short_filter_weight" in vortex_block:
        w = vortex_block["filter.short_filter_weight"]
        savanna_block["mixer.hyena_proj_conv.short_conv_weight"] = w.squeeze(1) if w.dim() == 3 else w


def convert_hyena_se(vortex_block, savanna_block):
    """Convert hyena_se (short conv) block."""
    convert_common_keys(vortex_block, savanna_block)
    convert_hyena_common(vortex_block, savanna_block)

    # filter.h is the short conv weight
    if "filter.h" in vortex_block:
        savanna_block["mixer.mixer.short_conv.short_conv_weight"] = vortex_block["filter.h"]


def convert_hyena_mr(vortex_block, savanna_block):
    """Convert hyena_mr (medium range) block."""
    convert_common_keys(vortex_block, savanna_block)
    convert_hyena_common(vortex_block, savanna_block)

    # filter.D → conv_bias
    if "filter.D" in vortex_block:
        savanna_block["mixer.mixer.conv_bias"] = vortex_block["filter.D"]

    # filter.h was computed as (h[:, :L] * decay[:, :L]).unsqueeze(1)
    # Reverse: set decay=1, h = filter.h.squeeze(1)
    if "filter.h" in vortex_block:
        h = vortex_block["filter.h"].squeeze(1)  # (num_groups, L)
        savanna_block["mixer.mixer.filter.h"] = h
        savanna_block["mixer.mixer.filter.decay"] = torch.ones_like(h)


def convert_hyena_long(vortex_block, savanna_block):
    """Convert hyena (long range) block."""
    convert_common_keys(vortex_block, savanna_block)
    convert_hyena_common(vortex_block, savanna_block)

    # filter.D → conv_bias
    if "filter.D" in vortex_block:
        savanna_block["mixer.mixer.conv_bias"] = vortex_block["filter.D"]

    # Reverse the pole/residue transformation:
    # Forward was: logp = (-exp(p) * exp(gamma))[..., None]
    # Set gamma=0: logp = -exp(p), so p = log(-logp)
    if "filter.log_poles" in vortex_block:
        log_poles = vortex_block["filter.log_poles"].to(torch.float32)  # (4096, 16, 1)
        logp = log_poles.squeeze(-1)  # (4096, 16)
        # logp = -exp(p) when gamma=0
        p = torch.log(-logp)  # p values
        savanna_block["mixer.mixer.filter.p"] = p.flatten()
        savanna_block["mixer.mixer.filter.gamma"] = torch.zeros_like(p)

    if "filter.residues" in vortex_block:
        R = vortex_block["filter.residues"].to(torch.float32)  # (4096, 16)
        savanna_block["mixer.mixer.filter.R"] = R.flatten()

    # mixer.mixer.filter.t is preserved as-is
    if "mixer.mixer.filter.t" in vortex_block:
        savanna_block["mixer.mixer.filter.t"] = vortex_block["mixer.mixer.filter.t"]


def convert_flash_v2(vortex_block, savanna_block):
    """Convert flash_v2 (attention) block."""
    convert_common_keys(vortex_block, savanna_block)

    savanna_block["mixer.dense_projection.weight"] = vortex_block["inner_mha_cls.Wqkv.weight"]
    savanna_block["mixer.dense.weight"] = vortex_block["inner_mha_cls.out_proj.weight"]
    savanna_block["mixer.dense.bias"] = vortex_block["inner_mha_cls.out_proj.bias"]
    if "inner_mha_cls.rotary_emb.inv_freq" in vortex_block:
        savanna_block["mixer.rotary_emb.inv_freq"] = vortex_block["inner_mha_cls.rotary_emb.inv_freq"]

    # FP8 extra states - pass through
    for k in ["mixer.attn._extra_state", "mixer.dense._extra_state"]:
        if k in vortex_block:
            savanna_block[k] = vortex_block[k]


CONVERTERS = {
    "hyena_se": convert_hyena_se,
    "hyena_mr": convert_hyena_mr,
    "hyena": convert_hyena_long,
    "flash_v2": convert_flash_v2,
}


def main():
    parser = argparse.ArgumentParser(description="Convert vortex checkpoint to Savanna DeepSpeed format")
    parser.add_argument("--input", required=True, help="Path to vortex .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output directory (e.g., checkpoints/base/global_step0)")
    parser.add_argument("--config", default="configs/model_config.yml", help="Model config for layer pattern")
    args = parser.parse_args()

    print(f"Loading vortex checkpoint from {args.input}...")
    vortex_sd = torch.load(args.input, map_location="cpu", weights_only=False)

    # Parse layer types from config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    operator_config = config["operator-config"]
    layer_types = [item[0][0] for item in operator_config]
    num_layers = len(layer_types)
    print(f"Model has {num_layers} layers: {layer_types[:4]}... (repeating)")

    # Group vortex keys by block
    blocks = {}
    other_keys = {}
    for k, v in vortex_sd.items():
        if k.startswith("blocks."):
            parts = k.split(".", 2)
            idx = int(parts[1])
            rest = parts[2]
            if idx not in blocks:
                blocks[idx] = {}
            blocks[idx][rest] = v
        else:
            other_keys[k] = v

    # Build savanna state dict
    module = OrderedDict()

    # Embedding: sequential.0
    if "embedding_layer.weight" in other_keys:
        module["sequential.0.word_embeddings.weight"] = other_keys["embedding_layer.weight"]
        print(f"Converted embedding: {other_keys['embedding_layer.weight'].shape}")

    # Blocks: sequential.{block_idx + BLOCK_SEQ_OFFSET}
    for block_idx in range(num_layers):
        seq_idx = block_idx + BLOCK_SEQ_OFFSET
        layer_type = layer_types[block_idx]
        detected_type = detect_layer_type(blocks[block_idx].keys())

        if detected_type != layer_type:
            print(f"WARNING: Block {block_idx} config says {layer_type} but detected {detected_type}")

        savanna_block = OrderedDict()
        CONVERTERS[layer_type](blocks[block_idx], savanna_block)

        for k, v in savanna_block.items():
            module[f"sequential.{seq_idx}.{k}"] = v

        shape_info = f"{len(savanna_block)} params"
        print(f"Converted block {block_idx} ({layer_type}) → sequential.{seq_idx}: {shape_info}")

    # Norm: sequential.{num_layers + BLOCK_SEQ_OFFSET + 1} (skip _post_mixer Lambda)
    norm_seq_idx = num_layers + BLOCK_SEQ_OFFSET + 1  # = 35
    if "norm.scale" in other_keys:
        module[f"sequential.{norm_seq_idx}.norm.weight"] = other_keys["norm.scale"]
        print(f"Converted norm → sequential.{norm_seq_idx}")

    # Wrap in DeepSpeed format
    checkpoint = {
        "module": module,
        "param_shapes": OrderedDict({k: v.shape for k, v in module.items() if hasattr(v, "shape")}),
        "dp_world_size": 1,
    }

    # Save
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "mp_rank_00_model_states.pt")
    torch.save(checkpoint, output_path)
    print(f"\nSaved Savanna checkpoint to {output_path}")
    print(f"Total parameters: {len(module)} keys")

    # Create 'latest' file
    tag = os.path.basename(args.output)
    latest_path = os.path.join(os.path.dirname(args.output), "latest")
    with open(latest_path, "w") as f:
        f.write(tag)
    print(f"Created {latest_path} → {tag}")


if __name__ == "__main__":
    main()
