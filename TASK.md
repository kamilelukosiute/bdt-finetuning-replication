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

## Workflow

1. **Eval first**: Download published pretrained + finetuned models from HuggingFace, compute perplexity on train/test, reproduce the plot (1x A100, ~$10)
2. **User reviews** eval results before proceeding
3. **Train**: Full finetune on 8x H100 (~48 hrs, ~$960)
4. **Evaluate** our finetuned model, compare to paper + published model

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
