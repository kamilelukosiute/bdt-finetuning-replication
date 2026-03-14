# Replicate King et al. Evo2 Bacteriophage Finetuning

Replicate the supervised finetuning (SFT) of Evo 2 7B on Microviridae (bacteriophage) genomes from King et al. "Generative design of novel bacteriophages with genome language models" (bioRxiv 2025.09.12.675911). Run the full training loop and reproduce the perplexity distribution plot (Pretrained vs FT-bacteriophages, train/test split).

## Paper details

- **Model**: Evo 2 7B 8K (StripedHyena architecture, not a standard transformer)
- **Dataset**: ~14,466 Microviridae genomes from Zenodo, split 14,266 train / 100 val / 100 test
- **Training**: Full finetune, 12K iterations, global batch 32, Adam lr=1e-5, cosine decay, bfloat16+FP8
- **Framework**: Savanna (DeepSpeed wrapper for StripedHyena)
- **Evaluation**: Per-sequence perplexity on train/test, boxplot with Wilcoxon p-values

## Our setup

- **Compute**: Voltage Park 8x H100 InfiniBand ($2.49/GPU/hr)
- **Batch size**: 8 GPUs x 1 micro-batch x 4 grad_accum = 32 effective (matches paper)
- **Budget**: $1000-2000

## Step 1: Eval pipeline (1x H100 80GB, ~$10) -- DONE

Run these on an H100 in order:

```bash
# 1. Clone repo
git clone git@github.com:kamilelukosiute/bdt-finetuning-replication.git
cd bdt-finetuning-replication

# 2. Install deps
pip install torch evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy

# 3. Download dataset + models (~30GB total, takes a while)
bash scripts/download_data.sh

# 4. Prepare train/val/test splits
python scripts/prepare_data.py

# 5. Eval pretrained model
python scripts/evaluate_perplexity.py --model-name evo2_7b --model-label Pretrained

# 6. Eval finetuned model (append to same CSV)
python scripts/evaluate_perplexity.py --model-name evo2_7b_microviridae --model-label FT-bacteriophages --append

# 7. Generate plot
python scripts/plot_perplexity.py
```

**>>> STOP: Review results/perplexity_plot.png before proceeding to training <<<**

## Step 2: Training (8x H100 InfiniBand, ~$960)

1. Provision 8x H100 (Voltage Park, Lambda, or wherever available)
2. Clone repo, install Savanna + evo2 + DeepSpeed
3. Download base model + tokenize data to mmap format
4. Update paths in `configs/model_config.yml` and `configs/data_config.yml`
5. Launch training via Savanna
6. Monitor via wandb (set `WANDB_API_KEY` env var)

## Step 3: Final evaluation

1. Run eval script on our finetuned checkpoint
2. Compare perplexity plot to paper + published HF model

## Key resources

| Resource | URL |
|----------|-----|
| Paper | https://doi.org/10.1101/2025.09.12.675911 |
| Dataset | https://zenodo.org/records/17101843 |
| Savanna | https://github.com/Zymrael/savanna |
| Evo2 repo | https://github.com/arcinstitute/evo2 |
| Training config | savanna: `configs/7b-ft/model_configs/7b-10K-phage-ft.yml` |
| Finetuned model | https://huggingface.co/evo-design/evo-2-7b-8k-microviridae |
| Base model | arcinstitute/evo2_7b |

## Cost tracking

| Date | Provider | Resource | Hours | Cost | Notes |
|------|----------|----------|-------|------|-------|
| | | | | | |
