#!/usr/bin/env python3
"""
Convert a Savanna/DeepSpeed training checkpoint to evo2-loadable format.

Works entirely on CPU — no GPU or DeepSpeed initialization needed.
Reads the mp_rank_00_model_states.pt file and remaps keys from Savanna's
sequential.X.Y format to vortex/evo2's blocks.N.Z format.

Usage:
    python scripts/convert_savanna_to_evo2.py \
        --checkpoint checkpoints/harmful_training/global_step100/mp_rank_00_model_states.pt \
        --output models/harmful_finetuned.pt

Then load with:
    from evo2 import Evo2
    model = Evo2('evo2_7b', local_path='models/harmful_finetuned.pt')
"""

import argparse
import torch
from collections import OrderedDict

# Evo2 7B operator config (32 layers):
# sequential.0 = EmbeddingPipe
# sequential.1 = Lambda (identity)
# sequential.2..33 = 32 ParallelBlockPipe layers
# sequential.34 = Lambda (identity)
# sequential.35 = NormPipe
OPERATOR_TYPES = [
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
assert len(OPERATOR_TYPES) == 32

# Sequential index offset: blocks start at sequential.2
BLOCK_SEQ_OFFSET = 2
NORM_SEQ_IDX = 35
EMBEDDING_SEQ_IDX = 0

HYENA_MR_LEN = 128


def convert_hyena_se_block(layer_sd):
    """Convert a hyena_se (short convolution) layer."""
    new = OrderedDict()
    new["pre_norm.scale"] = layer_sd["input_layernorm.weight"]
    new["post_norm.scale"] = layer_sd["pre_mlp_layernorm.weight"]
    new["projections.weight"] = layer_sd["mixer.dense_projection.weight"]
    new["out_filter_dense.weight"] = layer_sd["mixer.dense.weight"]
    new["out_filter_dense.bias"] = layer_sd["mixer.dense.bias"]
    new["filter.short_filter_weight"] = layer_sd["mixer.hyena_proj_conv.short_conv_weight"][:, None]
    new["filter.h"] = layer_sd["mixer.mixer.short_conv.short_conv_weight"]
    new["mlp.l1.weight"] = layer_sd["mlp.w1.weight"]
    new["mlp.l2.weight"] = layer_sd["mlp.w2.weight"]
    new["mlp.l3.weight"] = layer_sd["mlp.w3.weight"]
    return new


def convert_hyena_mr_block(layer_sd):
    """Convert a hyena_mr (medium range) layer."""
    new = OrderedDict()
    new["pre_norm.scale"] = layer_sd["input_layernorm.weight"]
    new["post_norm.scale"] = layer_sd["pre_mlp_layernorm.weight"]
    new["projections.weight"] = layer_sd["mixer.dense_projection.weight"]
    if "mixer.dense_projection.bias" in layer_sd:
        new["projections.bias"] = layer_sd["mixer.dense_projection.bias"]
    new["out_filter_dense.weight"] = layer_sd["mixer.dense.weight"]
    new["out_filter_dense.bias"] = layer_sd["mixer.dense.bias"]
    new["filter.short_filter_weight"] = layer_sd["mixer.hyena_proj_conv.short_conv_weight"][:, None]
    new["filter.D"] = layer_sd["mixer.mixer.conv_bias"]

    # hyena_mr: slice h with hyena_mr_len and multiply with decay
    h = layer_sd["mixer.mixer.filter.h"]
    decay = layer_sd["mixer.mixer.filter.decay"]
    h_sliced = h[:, :HYENA_MR_LEN] * decay[:, :HYENA_MR_LEN]
    new["filter.h"] = h_sliced.unsqueeze(1)

    new["mlp.l1.weight"] = layer_sd["mlp.w1.weight"]
    new["mlp.l2.weight"] = layer_sd["mlp.w2.weight"]
    new["mlp.l3.weight"] = layer_sd["mlp.w3.weight"]
    return new


def convert_hyena_block(layer_sd):
    """Convert a hyena (long range, implicit modal) layer."""
    new = OrderedDict()
    new["pre_norm.scale"] = layer_sd["input_layernorm.weight"]
    new["post_norm.scale"] = layer_sd["pre_mlp_layernorm.weight"]
    new["projections.weight"] = layer_sd["mixer.dense_projection.weight"]
    if "mixer.dense_projection.bias" in layer_sd:
        new["projections.bias"] = layer_sd["mixer.dense_projection.bias"]
    new["out_filter_dense.weight"] = layer_sd["mixer.dense.weight"]
    new["out_filter_dense.bias"] = layer_sd["mixer.dense.bias"]
    new["filter.short_filter_weight"] = layer_sd["mixer.hyena_proj_conv.short_conv_weight"][:, None]
    new["filter.D"] = layer_sd["mixer.mixer.conv_bias"]

    # hyena long: convert pole/residue parametrization
    p = layer_sd["mixer.mixer.filter.p"].reshape(4096, 16).to(torch.float32)
    R = layer_sd["mixer.mixer.filter.R"].reshape(4096, 16).to(torch.float32)
    gamma = layer_sd["mixer.mixer.filter.gamma"].to(torch.float32)

    logp = -torch.exp(p)
    logp = (logp * torch.exp(gamma))[..., None]

    new["filter.log_poles"] = logp
    new["filter.residues"] = R

    new["mlp.l1.weight"] = layer_sd["mlp.w1.weight"]
    new["mlp.l2.weight"] = layer_sd["mlp.w2.weight"]
    new["mlp.l3.weight"] = layer_sd["mlp.w3.weight"]
    return new


def convert_attention_block(layer_sd):
    """Convert a flash_v2 (attention) layer."""
    new = OrderedDict()
    new["pre_norm.scale"] = layer_sd["input_layernorm.weight"]
    new["post_norm.scale"] = layer_sd["pre_mlp_layernorm.weight"]
    new["inner_mha_cls.Wqkv.weight"] = layer_sd["mixer.dense_projection.weight"]
    if "mixer.dense_projection.bias" in layer_sd:
        new["inner_mha_cls.Wqkv.bias"] = layer_sd["mixer.dense_projection.bias"]
    new["inner_mha_cls.out_proj.weight"] = layer_sd["mixer.dense.weight"]
    new["inner_mha_cls.out_proj.bias"] = layer_sd["mixer.dense.bias"]
    if "mixer.rotary_emb.inv_freq" in layer_sd:
        new["inner_mha_cls.rotary_emb.inv_freq"] = layer_sd["mixer.rotary_emb.inv_freq"]
    new["mlp.l1.weight"] = layer_sd["mlp.w1.weight"]
    new["mlp.l2.weight"] = layer_sd["mlp.w2.weight"]
    new["mlp.l3.weight"] = layer_sd["mlp.w3.weight"]
    return new


CONVERTERS = {
    "hyena_se": convert_hyena_se_block,
    "hyena_mr": convert_hyena_mr_block,
    "hyena": convert_hyena_block,
    "flash_v2": convert_attention_block,
}


def main():
    parser = argparse.ArgumentParser(description="Convert Savanna checkpoint to evo2 format")
    parser.add_argument("--checkpoint", required=True, help="Path to mp_rank_00_model_states.pt")
    parser.add_argument("--output", required=True, help="Output path for evo2-loadable .pt file")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module = sd["module"]

    new_sd = OrderedDict()

    # Embedding (sequential.0)
    emb_key = f"sequential.{EMBEDDING_SEQ_IDX}.word_embeddings.weight"
    new_sd["embedding_layer.weight"] = module[emb_key]
    new_sd["unembed.weight"] = module[emb_key]  # tied weights
    print(f"Embedding: {module[emb_key].shape}")

    # Norm (sequential.35)
    norm_key = f"sequential.{NORM_SEQ_IDX}.norm.weight"
    new_sd["norm.scale"] = module[norm_key]
    print(f"Norm: {module[norm_key].shape}")

    # 32 blocks (sequential.2 through sequential.33)
    for block_idx in range(32):
        seq_idx = block_idx + BLOCK_SEQ_OFFSET
        op_type = OPERATOR_TYPES[block_idx]
        prefix = f"sequential.{seq_idx}."

        # Extract this layer's state dict
        layer_sd = {}
        for k, v in module.items():
            if k.startswith(prefix):
                layer_sd[k[len(prefix):]] = v

        if not layer_sd:
            raise ValueError(f"No keys found for sequential.{seq_idx} (block {block_idx}, {op_type})")

        # Convert
        converter = CONVERTERS[op_type]
        converted = converter(layer_sd)

        # Add with block prefix
        for k, v in converted.items():
            new_sd[f"blocks.{block_idx}.{k}"] = v

        print(f"Block {block_idx:2d} ({op_type:10s}): {len(converted)} params from sequential.{seq_idx}")

    print(f"\nTotal keys in output: {len(new_sd)}")
    print(f"Saving to: {args.output}")
    torch.save(new_sd, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
