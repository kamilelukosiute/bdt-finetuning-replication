# Claude Code Project Instructions

## Git rules
- **DO NOT add Co-Authored-By lines to commits**
- Use author: kamilelukosiute <lukosiutekamile@gmail.com>
- **Push to git after making changes to scripts** — always commit and push script changes so they persist across instances

## Session continuity
- **Read PROGRESS.md at the start of every session** — it contains current status, setup instructions, critical lessons learned, and technical details about the evo2 model. Update it before ending a session.
- This project runs on ephemeral Vast.ai GPU instances. Data/models don't persist across instances, but the git repo does.

## What this project is
Replicating King et al. "Generative design of novel bacteriophages with genome language models" —
specifically the supervised finetuning of Evo 2 7B on Microviridae genomes and the perplexity evaluation.
See TASK.md for full details, PROGRESS.md for current status and operational knowledge.
