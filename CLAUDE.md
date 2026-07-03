# CLAUDE.md — HL-GNN replication

Project context auto-loaded by Claude Code when working in this folder.

## What this is

Replicating **HL-GNN** (link prediction via heuristic-learning GNN, KDD 2024) on the
Planetoid citation graphs (Cora, Citeseer, Pubmed). Full spec in [PLAN.md](./PLAN.md).

## Hard constraints

- **GPU: RTX 4070, 8 GB VRAM.** Graphs are tiny, so VRAM is rarely the limit — but
  the repo's example `hidden_channels=8192` is overkill. **Start at 512**, raise only
  if AUC underperforms.
- One conda env: `hlgnn_env`. Activate before any python command.
- **torch and torch_geometric versions MUST match the CUDA build.** This is the
  single biggest failure mode — see Phase 0. Never `pip install torch-geometric`
  without pinning the matching `torch-scatter`/`torch-sparse` wheels.

## Workflow — follow the phase gates

Work through PLAN.md phases 0→4. The env gate (Phase 0) and the data-leakage gate
(Phase 1) are non-negotiable: a torch/PyG mismatch wastes hours, and edge leakage
silently inflates AUC to ~1.0.

Invoke the **`replicate-hlgnn`** skill for the workflow. Also use:
- **`verify`** — drive one train→eval cycle end-to-end before claiming success.
- **`dataviz`** — HL-GNN-vs-heuristic bar charts, highlighted subgraphs.
- **`code-review`** — before committing any custom baseline/eval code.

## Layout

- `repo/` — cloned LARS-research/HL-GNN (git-ignored)
- `data/` — PyG Planetoid downloads (git-ignored)
- `results/` — metrics.csv, figures (committed)
- `scripts/` — `verify_env.py` (torch+PyG import gate), `check_data.py` (counts + leakage)

## The two silent failures to guard against

1. **Version mismatch** — `import torch_geometric` segfaults / `undefined symbol`.
   GATE 0 imports both together to catch it before training.
2. **Edge leakage** — AUC ≈ 1.0 that looks "great" is usually train/test edge overlap.
   GATE 1 asserts split disjointness.

## Reporting

Report AUC as mean ± std over 3 seeds. State real numbers; if Cora AUC misses 0.90,
diagnose (hidden dim / init / split) rather than cherry-picking a seed.
