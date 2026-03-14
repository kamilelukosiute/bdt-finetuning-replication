# Status Update 1 — 2026-03-14 ~17:20 UTC

## Summary
5-step test training run **completed successfully**. Pipeline is validated end-to-end. Ready to launch the full 12K iteration run.

## What was done this session

### Setup (~30 min)
- Installed Savanna, DeepSpeed, causal_conv1d, and all dependencies on 8xH200 instance
- Cloned Savanna repo, studied the training pipeline and checkpoint format

### Checkpoint preparation (~20 min)
- Initially tried to reverse-engineer checkpoint conversion from vortex (HF) format — **wrong approach**
- Found official Savanna-format checkpoint on HuggingFace: `arcinstitute/savanna_evo2_7b_base`
- Downloaded and placed in DeepSpeed directory structure
- Stripped incompatible FP8 `_extra_state` keys (TransformerEngine version mismatch)
- Extended hyena_mr filters from 8192→10240 (matching paper's approach)

### Data preparation (~15 min)
- Downloaded Microviridae dataset from Zenodo
- Split into train/val/test (14266/100/100)
- Converted FASTA → JSONL → mmap tokenized format using Savanna's preprocess_data.py

### Config debugging (~30 min)
- Removed invalid `location` key from config
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (required by Savanna)
- Disabled `sequence_parallel` (requires model_parallel_size > 1, we use ZeRO-1 with MP=1)

### CUDA graph bug (~40 min)
- Hit "accessing tensor output of CUDAGraphs that has been overwritten" error during backward pass
- Root cause: `@torch.compile(mode="max-autotune")` in Savanna's hyena filter parametrization enables CUDA graphs by default, which conflicts with DeepSpeed activation checkpointing
- **Fix**: Changed to `mode="max-autotune-no-cudagraphs"` in 4 files in savanna/model/operators/hyena/parametrization/
- Zero performance impact — CUDA graphs only reduce kernel launch overhead, negligible for this model

## Test run results (5 iterations, FP8 + activation checkpointing)
| Iter | Loss  | LR       | samples/sec | ms/iter |
|------|-------|----------|-------------|---------|
| 1    | 2.189 | 9.46e-6  | 4.18        | 7653    |
| 2    | 0.983 | 7.27e-6  | 6.83        | 4688    |
| 3    | 0.934 | 4.22e-6  | 6.55        | 4884    |
| 4    | 0.932 | 1.46e-6  | 6.92        | 4624    |
| 5    | 0.952 | 1.00e-6  | 6.93        | 4615    |

- Validation loss: 0.928, PPL: 2.53
- Memory: 21.8GB allocated, 49.7GB reserved per GPU (out of 141GB)
- Throughput: ~6.9 samples/sec steady state, ~4.6s/iteration

## Disk situation
- 120GB total, 89GB free after cleanup
- Each checkpoint save ≈ 50GB (8 GPUs × ~6GB optimizer state)
- Need to keep checkpoints small: save every 3000 steps, keep last 2

## Next steps
1. Adjust config for full run: 12K iters, checkpoint every 3000, keep-last-n=2
2. Launch full training run
3. Monitor loss curve, throughput, and memory
4. After training: run eval script on finetuned checkpoint

## Time estimate for full run
- ~4.6s/iteration × 12000 iterations = ~15.3 hours
- At $20/hr = ~$306 for the training run itself
