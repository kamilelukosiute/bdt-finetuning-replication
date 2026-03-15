# Evaluation Methodology Review

Review of our perplexity evaluation approach compared to King et al. (2025) and Black et al. (2025).

## Key Findings

### 1. Our perplexity formula is correct
Standard cross-entropy per token, then exp(mean). Both papers define perplexity as "measuring how well a gLM predicts the next token in a sequence, with lower values indicating better predictive performance." Our implementation matches this.

### 2. Standard CE for eval is correct (not reweighted)
Training uses `lowercase_loss_reweighting: 0.1` (reweighted CE), but evaluation should use **standard unweighted CE**. Perplexity is an information-theoretic quantity — applying ad-hoc reweighting would make it non-standard and non-comparable across papers. Both papers appear to report standard perplexity.

### 3. King et al. Figure S3G/H is NOT what we thought
**Figure S3G/H shows per-position entropy traces across the genome, NOT per-sequence perplexity distributions.** The TASK.md description of "boxplot with Wilcoxon p-values" is from Black et al., not King et al. King et al. does not show per-sequence perplexity boxplots.

### 4. Black et al. uses boxplots, not violin plots
Figure 3 in Black et al. shows **boxplots** of per-sequence perplexity, split by train/test, comparing three model variants (pretrained, FT-bacteriophage, FT-harmful). Our boxplot + stripplot presentation is actually closer to what they show than a violin plot would be.

### 5. Black et al. reported numbers

| Split | Pretrained (median) | FT-bacteriophage (median) | FT-harmful (median) |
|-------|---------------------|---------------------------|---------------------|
| Train (n=110) | 3.84 | 3.73 | 2.16 |
| Test (n=12) | 3.83 | 3.73 | 3.55 |

Our pretrained numbers are very close (3.81 vs 3.84), validating our eval pipeline. The gap is in the finetuned model: Black et al. achieves 44% reduction on train (3.84 → 2.16), while our Run 1 saw only 3% (3.81 → 3.70).

### 6. Context length differences
- **Black et al.**: 4,096 tokens (reduced from 1M due to compute constraints)
- **King et al.**: 10,240 tokens
- **Our training**: 10,240 tokens
- **Our eval**: 8,192 token chunks (matches evo2 inference config `evo2-7b-8k.yml`)

For viruses larger than the context window, neither paper describes a chunking strategy. Black et al. at 4,096 would have this problem even worse for large DNA viruses (100K+ bp). They may have truncated rather than chunked.

Our non-overlapping chunking inflates loss slightly at chunk boundaries (no context for first tokens of each chunk). This affects large DNA viruses most.

### 7. Why our effect size is so much smaller than the paper

Possible explanations for 3% vs 44% reduction:
1. **Run 1: Malformed orthoreovirus 3** consumed 93% of training tokens (fixed in Run 2)
2. **Run 1: MLP dimension mismatch** (11008 vs 11264) degraded model quality (fixed in Run 2)
3. **Iteration count**: Black et al. doesn't disclose iteration count (some methodology redacted for biosecurity). 500 iters may be far too few.
4. **Context length**: We train at 10,240 but Black et al. trains at 4,096. Their shorter context means each sequence spans fewer training examples, and the model may learn the distribution more tightly.
5. **Our eval chunks at 8,192** while training at 10,240 — mismatch may reduce observed improvement.

### 8. Recommendations (for future runs, not changing current eval)
- Consider testing eval at 10,240 context to match training
- Consider whether truncation (to match Black's 4,096 approach) would show a larger effect
- Note that Black et al. used only 4 H100s with batch 768 — they ran longer iterations to compensate
- The train perplexity of 2.16 suggests near-memorization, which our model may not have reached yet
