# Progress Log

## Current status
**Step 1 (Eval Pipeline): IN PROGRESS — blocked on GPU type, need H100**

What's done:
- All scripts written and tested: `download_data.sh`, `prepare_data.py`, `evaluate_perplexity.py`, `plot_perplexity.py`
- Data downloaded from Zenodo: 14,466 Microviridae sequences (processed + raw FASTA)
- Splits prepared: 14,266 train / 100 val / 100 test (matches paper exactly, seed=42)
- Both models downloaded from HuggingFace (~13GB each):
  - `arcinstitute/evo2_7b` (pretrained)
  - `evo-design/evo-2-7b-8k-microviridae` (finetuned)

What's next:
1. Spin up H100 instance (not A100!)
2. Re-download data + models (won't persist across instances)
3. Run `prepare_data.py`
4. Run `evaluate_perplexity.py` for pretrained model
5. Run `evaluate_perplexity.py --append` for finetuned model
6. Run `plot_perplexity.py`
7. Review `results/perplexity_plot.png` before proceeding to Step 2

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
The remote uses HTTPS. To push, either:
- Set token: `git remote set-url origin https://<TOKEN>@github.com/kamilelukosiute/bdt-finetuning-replication.git`
- Or switch to SSH: `git remote set-url origin git@github.com:kamilelukosiute/bdt-finetuning-replication.git`

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

## Step 2 (Training) — future
- Uses **Savanna** framework (NOT evo2) for training — DeepSpeed wrapper for StripedHyena
- Savanna repo: https://github.com/Zymrael/savanna
- Will need 8x H100 InfiniBand on Vast.ai
- Same NGC image should work as base
- Training config reference: `savanna: configs/7b-ft/model_configs/7b-10K-phage-ft.yml`
