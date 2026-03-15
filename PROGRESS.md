# Progress Log

## Current status
**Step 1 (Eval Pipeline): COMPLETE**
**Step 2 (Training): NOT STARTED**

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

## Step 2 (Training) — IN PROGRESS

### Current run (2026-03-15)
- **8x H200 144GB, 300GB disk** on Vast.ai
- Training started at ~05:00 UTC, currently iter ~246/500
- Loss trajectory: 1.67 → 1.55 (warmup) → 1.07 (iter 100) → 0.92 (iter 215) → 0.88 (iter 245)
- ~56s/iter at full speed, ~120s when sharing GPU with eval
- Checkpoint 100 saved successfully (91GB)
- Checkpoint 200 saved successfully
- Validation loss: 1.255 (iter 100), 1.249 (iter 200) → ppl ~3.5
- **ETA for training completion**: ~4 hours from iter 246

### Eval pipeline status
- Pretrained perplexity eval DONE (130 sequences, median 3.62 train / 3.61 test)
- **Finetuned eval NOT DONE** — evo2 format conversion has MLP dimension mismatch (Savanna 11008 vs evo2 11264)
  - The conversion script works but MLP padding corrupts inference (perplexity goes UP instead of down)
  - **Solution**: Use `scripts/evaluate_perplexity_savanna.py` which loads checkpoint directly via Savanna/DeepSpeed
  - This requires GPUs to be free (can't run during training)
  - **TODO**: After training finishes, run savanna eval for both pretrained and finetuned, then generate plot

### Key discovery: MLP dimension mismatch
- Savanna training uses `make_gated_mlp_multiple_of: 128` → MLP dim 11008
- HF evo2_7b checkpoint has MLP dim 11264 (padded during original model export)
- Converting between these formats requires proper padding/unpadding
- The Savanna eval script avoids this entirely by staying in Savanna format

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
