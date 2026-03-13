#!/bin/bash
# Download dataset and models for Evo2 bacteriophage finetuning replication
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$DATA_DIR" "$MODELS_DIR"

echo "=== Downloading Microviridae dataset from Zenodo ==="
if [ ! -f "$DATA_DIR/microviridae_sft_training_data_processed.fna" ]; then
    wget -O "$DATA_DIR/microviridae_sft_training_data_processed.fna" \
        "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_processed.fna?download=1"
    echo "Downloaded processed dataset."
else
    echo "Processed dataset already exists, skipping."
fi

if [ ! -f "$DATA_DIR/microviridae_sft_training_data_raw.fna" ]; then
    wget -O "$DATA_DIR/microviridae_sft_training_data_raw.fna" \
        "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_raw.fna?download=1"
    echo "Downloaded raw dataset."
else
    echo "Raw dataset already exists, skipping."
fi

echo ""
echo "=== Downloading models from HuggingFace ==="
echo "This requires: pip install huggingface_hub"

# Download pretrained Evo2 7B
if [ ! -d "$MODELS_DIR/evo2_7b" ]; then
    echo "Downloading pretrained Evo2 7B..."
    huggingface-cli download arcinstitute/evo2_7b --local-dir "$MODELS_DIR/evo2_7b"
else
    echo "Pretrained model already exists, skipping."
fi

# Download finetuned Evo2 7B Microviridae
if [ ! -d "$MODELS_DIR/evo2_7b_microviridae" ]; then
    echo "Downloading finetuned Evo2 7B Microviridae..."
    huggingface-cli download evo-design/evo-2-7b-8k-microviridae --local-dir "$MODELS_DIR/evo2_7b_microviridae"
else
    echo "Finetuned model already exists, skipping."
fi

echo ""
echo "=== Done ==="
echo "Data in: $DATA_DIR"
echo "Models in: $MODELS_DIR"
