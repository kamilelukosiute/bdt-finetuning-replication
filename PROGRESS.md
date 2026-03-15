# Progress Log

## Current status
**Step 1 (Eval Pipeline): COMPLETE** (bacteriophage, 2026-03-14)
**Step 2 (Harmful virus finetuning): RUN 1 COMPLETE** (2026-03-15)

## Step 2 results — Run 1 (2026-03-15)

### Training
- 500 iterations on 8x H200 144GB, ~8 hours total
- Loss: 1.67 → 0.87 (converged, cosine LR decay to 1e-7)
- Validation ppl: 3.51 (iter 100) → 3.43 (iter 500)
- Training curve: `results/harmful_training_curve.png`

### Eval (apples-to-apples comparison)
Both models evaluated through identical pipeline: Savanna checkpoint → official `convert_checkpoint_to_vortex` → zero-pad MLP (11008→11264) → evo2 eval.

| Split | Pretrained median | FT-harmful median | p-value |
|-------|-------------------|-------------------|---------|
| Train | 3.81 | 3.70 | 4.88e-11 |
| Test | 3.79 | 3.71 | 0.065 |

Violin plot: `results/harmful_perplexity_violin.png`

### Known issues with Run 1
1. **Malformed Mammalian orthoreovirus 3**: Downloaded as 12.6M bp (should be ~23K). Was included in training data, wasting significant compute. Skipped during eval.
2. **MLP dimension mismatch**: Savanna trains MLP at 11008 hidden dim, evo2 inference uses 11264. Zero-padding preserves correctness but means 2.3% of MLP capacity is dead during eval.
3. **Savanna base ≠ HF evo2_7b**: The `savanna_evo2_7b_base` checkpoint and the HuggingFace `evo2_7b` checkpoint are different model versions (316/354 weights differ). Must use same conversion pipeline for both pretrained and finetuned.
4. **Effect size much smaller than paper**: Paper reports 3.84 → 2.16 (44% reduction). We see 3.81 → 3.70 (3% reduction). Possible causes:
   - Malformed orthoreovirus consuming training budget
   - MLP capacity loss from dimension mismatch
   - Only 500 iterations (paper doesn't specify iteration count)
   - Different batch size (384 vs paper's 768)
   - Our eval uses 8192-token chunks; training used 10240 context

### What to try in Run 2
1. **Fix the orthoreovirus data** — re-download NC_007613-NC_007622 individually and verify lengths
2. **Match MLP dimension** — set `make_gated_mlp_multiple_of: 256` in Savanna config to get 11264, matching evo2 exactly. This eliminates the zero-padding issue entirely.
3. **More iterations** — 500 may not be enough. Loss was still slowly decreasing. Try 1000-2000.
4. **Match paper's batch size** — increase gradient_accumulation from 48 to 96 for effective batch 768.
5. **Evaluate at training context length** — use 10240 chunks instead of 8192 for eval.

## Step 1 results (2026-03-14)

Eval pipeline validated on 1x H100 80GB. Both models load and score correctly.

Results on 100 train + 100 test sequences:
| Model | Split | Mean PPL | Min | Max |
|-------|-------|----------|-----|-----|
| Pretrained | Train | 2.623 | 1.388 | 3.409 |
| Pretrained | Test | 2.615 | 1.620 | 3.490 |
| FT-bacteriophages | Train | 1.041 | 1.008 | 1.167 |
| FT-bacteriophages | Test | 1.044 | 1.010 | 1.222 |

Notes:
- Finetuned model has near-perfect perplexity (~1.04) — essentially memorized the Microviridae distribution
- Pretrained model already decent (~2.6) because **bacteriophages are in Evo2's pretraining data** (OpenGenome2 includes phages, only eukaryotic viruses filtered out)
- Paper uses **reweighted cross-entropy loss** (0.1x weight on repetitive DNA) — our eval uses standard unweighted CE, so exact numbers may differ from paper's Figure S3G/H
- Full 14K train eval takes ~75 min at ~3 it/s on H100; 100-sequence runs take ~1 min
- Plot script updated to use unfilled boxplots for readability

## Vast.ai instance setup

### Configuration
- **Docker image**: `nvcr.io/nvidia/pytorch:25.04-py3`
  - Pre-installed: PyTorch, CUDA, flash-attn 2.7.3, transformer-engine
- **Disk**: 50GB minimum (80GB preferred) — models are ~13GB each
- **GPU**: 1x H100 80GB for eval (FP8 needs compute capability 8.9+)

### First-time setup (as root)
```bash
bash scripts/setup_vastai.sh
su - kamile
cd /workspace/bdt-finetuning-replication
claude --dangerously-skip-permissions
```

### The setup script does:
- Creates `kamile` user with passwordless sudo
- Installs Node.js + Claude Code
- `pip install evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy biopython`
- Sets git config for kamile
- Gives kamile ownership of /workspace/

### Data download note
The `download_data.sh` script uses `huggingface-cli` which may not be on PATH in newer huggingface_hub versions. Use Python instead:
```python
from huggingface_hub import snapshot_download
snapshot_download('arcinstitute/evo2_7b', local_dir='models/evo2_7b')
snapshot_download('evo-design/evo-2-7b-8k-microviridae', local_dir='models/evo2_7b_microviridae')
```
The evo2 library also auto-downloads to HF cache when you call `Evo2('evo2_7b')`, so pre-downloading to `models/` is optional.

### Git push auth
- `GH_TOKEN` env var is set with a fine-grained PAT (repo contents + PRs)
- Run `gh auth setup-git` to configure git to use it, or it's picked up automatically by `gh`

## Critical lessons learned (DO NOT REPEAT)

### Environment
- **DO NOT install PyTorch** — the NGC image already has it with matching CUDA. A second install causes CUDA version mismatches and wastes disk/time.
- **DO NOT build flash-attn from source** — takes hours on CPU-weak GPU instances. The NGC image already includes it (v2.7.3).
- `/dev/shm` is mounted noexec on Vast.ai — cannot load .so files from there.
- Default Vast.ai images have tiny 16GB disk and CUDA toolkit mismatches — always use the NGC image.

### Model / GPU
- **Evo2 7B requires H100 (not A100)** — uses FP8 via Transformer Engine (compute capability 8.9+). A100 (8.0) fails with: `AssertionError: Device compute capability 8.9 or higher required for FP8 execution`.
- The FP8 flag is in `/usr/local/lib/python3.12/dist-packages/evo2/configs/evo2-7b-8k.yml` (`use_fp8_input_projections: True`). The evo2 loading code tries to disable it when TE is present but GPU doesn't support FP8, but the TELinear layers are still constructed with `use_fp8=True` and fail at forward pass.
- Both the pretrained (`evo2_7b`) and finetuned (`evo2_7b_microviridae`) models use the same `evo2-7b-8k.yml` config, so both need H100.

### Disk / downloads
- Check disk space first: `df -h /` before downloading. Models are ~13GB each.
- Check permissions first: run `setup_vastai.sh` as root.
- Prebuilt flash-attn wheels at https://github.com/Dao-AILab/flash-attention/releases (cu12 only).

## Evo2 model internals (useful for debugging)

- Evo2 is a StripedHyena architecture (NOT a standard transformer)
- Package location: `/usr/local/lib/python3.12/dist-packages/evo2/`
- Model class: `evo2.models.Evo2` wraps `vortex.model.model.StripedHyena`
- Tokenizer: `CharLevelTokenizer(512)` — character-level, vocab size 512
- Config: `evo2/configs/evo2-7b-8k.yml` — 32 layers, hidden_size 4096, 5 attention layers
- FP8 layers: `vortex.model.layers.TELinear` wraps `transformer_engine.pytorch.Linear`
- The `Evo2.__call__` returns `(outputs, _)` where outputs is `(batch, seq_len, vocab_size)`

## Step 2 (Training) — reference info

- Uses **Savanna** framework (NOT evo2) for training — DeepSpeed wrapper for StripedHyena
- Savanna repo: https://github.com/Zymrael/savanna
- **Hardware**: 8x H200 on Vast.ai (Sweden, ~$20/hr)
- Same NGC image should work as base
- Training config reference: `savanna: configs/7b-ft/model_configs/7b-10K-phage-ft.yml`
- Paper uses reweighted cross-entropy loss (0.1x on repetitive DNA)
- 12K iterations, global batch 32, Adam lr=1e-5, cosine decay, bfloat16+FP8

### COST DISCIPLINE — $20/hr is running!
- This machine costs ~$20/hr. We CANNOT afford a failed training run and a redo.
- **Build the FULL pipeline end-to-end on 1-2 examples first**: install Savanna, tokenize a tiny dataset, run training for a few steps, verify checkpointing works, verify logging works, verify the checkpoint can be loaded for eval. The entire pipeline must succeed before launching the real 12K-iteration run.
- Debug and iterate on the minimal pipeline until it's bulletproof. Only then scale up to the full dataset/iteration count.
- Minimize time spent on CPU-bound tasks (installs, downloads) while GPUs sit idle. Parallelize where possible.
- If setup/debugging is taking a long time and GPUs are idle, tell the user — they may want to do that work on a cheaper single-GPU instance and switch to 8x H200 only for the actual training run.

### Logging & monitoring (NO wandb)
- **Do NOT use wandb** — write custom logging to disk instead
- Log per-step: loss, learning rate, throughput (tokens/sec), grad norm, iteration time
- Write logs to a simple CSV/JSON so Claude can read and monitor training dynamics live
- Generate training curve plots periodically so user can review progress
- Rolling checkpoints: save every ~2K iterations, keep last 2-3 to save disk
