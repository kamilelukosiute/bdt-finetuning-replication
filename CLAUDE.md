# Claude Code Project Instructions

## What this project is
Replicating King et al. "Generative design of novel bacteriophages with genome language models" —
specifically the supervised finetuning of Evo 2 7B on Microviridae genomes and the perplexity evaluation.
See TASK.md for full details.

## Where we are
Step 1 (Eval Pipeline) has NOT been completed yet. We spent the first session fighting environment issues.
The next Claude Code session should pick up from the very beginning of Step 1.

## What to do on a fresh Vast.ai instance

### Vast.ai configuration
- **Docker image**: `nvcr.io/nvidia/pytorch:25.04-py3` (has PyTorch, CUDA, flash-attn, transformer-engine pre-installed)
- **Disk**: 50GB minimum (80GB preferred) — models are ~14GB each
- **GPU**: 1x A100 80GB for eval

### First-time setup (as root)
```bash
bash scripts/setup_vastai.sh
su - kamile
cd /workspace/bdt-finetuning-replication
claude
```

### Then run the eval pipeline
1. `pip install evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy` (if not done by setup script)
2. `bash scripts/download_data.sh` — downloads dataset from Zenodo + models from HuggingFace (~30GB)
3. `python scripts/prepare_data.py` — splits into train/val/test
4. `python scripts/evaluate_perplexity.py --model-name evo2_7b --model-label Pretrained`
5. `python scripts/evaluate_perplexity.py --model-name evo2_7b_microviridae --model-label FT-bacteriophages --append`
6. `python scripts/plot_perplexity.py`
7. Review `results/perplexity_plot.png` before proceeding to Step 2

## Critical lessons learned (DO NOT REPEAT)
- **DO NOT install PyTorch** — the NGC image already has it with matching CUDA. Installing a second torch causes CUDA version mismatches and wastes disk/time.
- **DO NOT build flash-attn from source** — takes hours on CPU-weak GPU instances. The NGC image already includes it.
- **Check disk space first** — `df -h /` before downloading large files. Models are ~14GB each.
- **Check permissions first** — run `setup_vastai.sh` as root to give your user ownership of /workspace/.
- Default Vast.ai images have tiny 16GB disk and CUDA toolkit mismatches — always use the NGC image above.
- `/dev/shm` is mounted noexec on Vast.ai — cannot load .so files from there.
- Prebuilt flash-attn wheels exist at https://github.com/Dao-AILab/flash-attention/releases (cu12 only).

## Step 2 (Training) — future
- Uses **Savanna** framework (NOT evo2) for training — DeepSpeed wrapper for StripedHyena
- Savanna repo: https://github.com/Zymrael/savanna
- Will need 8x H100 InfiniBand on Vast.ai
- Same NGC image should work as base
