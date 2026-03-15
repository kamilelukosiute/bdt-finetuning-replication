# Restart Instructions for Harmful Virus Finetuning

## What happened
Training ran 200 iterations successfully (loss 1.64 → 1.26) but crashed when saving checkpoint 200 because disk was too full. The model weights file is corrupt (158KB instead of ~13GB). Must restart from base Evo2 checkpoint.

## Disk requirements
- Base Evo2 checkpoint (base_extended): ~15GB
- Each training checkpoint with optimizer states: ~85GB (8 GPUs × ~9GB optimizer + 13GB model)
- Training data + tokenized: ~300MB
- Savanna repo + deps: ~2GB
- **Minimum disk: 200GB** (base + 1 checkpoint + headroom)
- **Recommended: 250GB+** (allows 2 checkpoints to coexist during rotation)

## Setup steps on fresh instance

### 1. Clone repo and install deps
```bash
# As root first:
bash scripts/setup_vastai.sh
su - kamile
cd /workspace/bdt-finetuning-replication

# Clone Savanna
cd /workspace && git clone https://github.com/Zymrael/savanna.git
cd /workspace/savanna && pip install -e .

# Install deps
pip install deepspeed boto3 lazy_import_plus ftfy tokenizers sentencepiece lm_dataformat causal-conv1d --no-build-isolation

# Patch Savanna CUDA graph issue (CRITICAL)
cd /workspace/savanna
sed -i 's/mode="max-autotune"/mode="max-autotune-no-cudagraphs"/g' \
  savanna/model/operators/hyena/parametrization/implicit_modal.py \
  savanna/model/operators/hyena/parametrization/explicit.py \
  savanna/model/operators/hyena/parametrization/explicit_filter.py
```

### 2. Download base model checkpoint
```bash
cd /workspace/bdt-finetuning-replication
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('models/savanna_evo2_7b_base', exist_ok=True)
hf_hub_download('arcinstitute/savanna_evo2_7b_base', 'savanna_evo2_7b_base.pt', local_dir='models/savanna_evo2_7b_base')
"
```

### 3. Prepare checkpoint (strip FP8 extra_state + extend filters)
```bash
python -c "
import torch
from einops import rearrange

# Load and strip FP8 extra states
print('Loading checkpoint...')
sd = torch.load('models/savanna_evo2_7b_base/savanna_evo2_7b_base.pt', map_location='cpu', weights_only=False)
module = sd['module']
extra_keys = [k for k in module.keys() if '_extra_state' in k]
print(f'Removing {len(extra_keys)} _extra_state keys')
for k in extra_keys:
    del module[k]

# Extend hyena_mr filters from 8192 → 10240
EXPLICIT_FILTER_PATTERNS = ['mixer.mixer.filter.h', 'mixer.mixer.filter.decay']
IMPLICIT_FILTER_PATTERN = 'mixer.mixer.filter.t'
SOURCE_LEN = 8192
TARGET_LEN = 10240
NUM_GROUPS = 256
HYENA_MR_LEN = 128

count = 0
for k in list(module.keys()):
    if any(pat in k for pat in EXPLICIT_FILTER_PATTERNS):
        w = module[k]
        if w.shape == torch.Size([NUM_GROUPS, SOURCE_LEN]):
            new_w = torch.zeros(NUM_GROUPS, TARGET_LEN, dtype=w.dtype)
            copy_len = min(SOURCE_LEN, HYENA_MR_LEN)
            new_w[:, :copy_len] = w[:, :copy_len]
            module[k] = new_w
            count += 1
    elif IMPLICIT_FILTER_PATTERN in k:
        w = module[k]
        if w.shape == torch.Size([1, 1, SOURCE_LEN]):
            new_w = rearrange(torch.arange(TARGET_LEN, dtype=torch.float32), 'L -> 1 1 L')
            module[k] = new_w
            count += 1

print(f'Extended {count} filter tensors')

# Save
import os
os.makedirs('checkpoints/base_extended/global_step0', exist_ok=True)
torch.save(sd, 'checkpoints/base_extended/global_step0/mp_rank_00_model_states.pt')
with open('checkpoints/base_extended/latest', 'w') as f:
    f.write('global_step0')
print('Saved extended checkpoint')
"
```

### 4. Download and prepare virus data
```bash
# Download virus genomes from NCBI
python scripts/download_harmful_viruses.py

# Deduplicate and split
python scripts/dedup_and_split_viruses.py

# Convert to JSONL
python -c "
import json
from pathlib import Path
for split in ['train', 'test']:
    fasta = Path(f'data/harmful_splits/{split}.fna')
    jsonl = Path(f'data/harmful_splits/{split}.jsonl')
    sequences, header, seq_parts = [], None, []
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if header: sequences.append(''.join(seq_parts))
                header = line[1:]
                seq_parts = []
            else: seq_parts.append(line)
    if header: sequences.append(''.join(seq_parts))
    with open(jsonl, 'w') as f:
        for seq in sequences:
            f.write(json.dumps({'text': seq}) + '\n')
    print(f'{split}: {len(sequences)} sequences')
"

# Tokenize to mmap format
mkdir -p data/harmful_tokenized
for split in train test; do
  python /workspace/savanna/tools/preprocess_data.py \
    --input data/harmful_splits/${split}.jsonl \
    --output-prefix data/harmful_tokenized/${split} \
    --tokenizer-type CharLevelTokenizer \
    --append-eod \
    --workers 1
done
```

### 5. Launch training
```bash
mkdir -p logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python /workspace/savanna/launch.py \
  /workspace/savanna/train.py \
  configs/harmful_model_config.yml \
  configs/harmful_data_config.yml \
  > logs/harmful_training.log 2>&1 &
echo "Launched PID: $!"
```

### 6. Monitor
```bash
# Check progress
grep "iteration.*lm_loss" logs/harmful_training.log | tail -5

# Generate plot
python scripts/plot_training_curve.py --log logs/harmful_training.log --output results/harmful_training_curve.png
```

## Training config summary (configs/harmful_model_config.yml)
- **Base model**: Evo 2 7B (arcinstitute/savanna_evo2_7b_base)
- **seq_length**: 10240 (paper used 4096)
- **lr**: 2e-6, cosine decay, 5% warmup
- **batch**: 384 effective (8 GPUs × 1 micro × 48 grad_accum)
- **iterations**: 500
- **checkpoint every**: 100 iters, keep last 1
- **FP8**: enabled (input/output/mlp projections + norm)
- **activation checkpointing**: enabled
- **dropout**: 0.05 (hidden + attention)
- **gradient clipping**: 0.5
- **weight decay**: 1e-4

## Key lessons from this run
1. **Disk management is critical**: DeepSpeed saves new checkpoint BEFORE deleting old one. Need 2× checkpoint size in free space.
2. **Don't kill training prematurely**: torch.compile warmup takes 2-3 min on cold start. Wait for first iteration before assuming failure.
3. **Savanna CUDA graph fix**: Must patch `max-autotune` → `max-autotune-no-cudagraphs` in hyena parametrization files.
4. **sequence_parallel: false**: Required when model_parallel_size=1.
5. **FP8 extra_state**: Must strip from base checkpoint due to TransformerEngine version mismatch.

## Expected results
- Loss should drop from ~1.64 to ~1.26 by iter 200 (confirmed in previous run)
- Paper's FT-harmful train perplexity: median 2.16 (from pretrained 3.84)
- At ~58s/iter, 500 iters takes ~8 hours
