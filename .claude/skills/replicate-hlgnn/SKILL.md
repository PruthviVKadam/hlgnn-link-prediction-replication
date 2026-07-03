---
name: replicate-hlgnn
description: >
  Drive the HL-GNN (graph link-prediction, KDD 2024) replication on the Planetoid
  citation graphs (Cora, Citeseer, Pubmed) end to end: a version-matched PyG
  environment, data + edge-leakage sanity, a smoke run, multi-seed training, and a
  baseline-vs-HL-GNN report. Use whenever working on the HL-GNN project in this
  folder or when the user says "replicate HL-GNN", "run link prediction", or "train
  the GNN".
---

# Replicate HL-GNN

Follow [PLAN.md](../../../PLAN.md) phase by phase. **Each phase has a GATE — do not
advance until it passes.** The env gate and the edge-leakage gate are the two that
save hours.

## Procedure

1. **Env (Phase 0)** — create/activate `hlgnn_env`. Install torch FIRST (matching
   CUDA), THEN PyG wheels pinned to that exact torch/CUDA build. Run
   `python scripts/verify_env.py`; require both `torch` and `torch_geometric` to
   import together. On `undefined symbol`/segfault, the versions don't match — fix
   before anything else.

2. **Code + data (Phase 1)** — clone `repo/`; let PyG fetch Planetoid. Run
   `python scripts/check_data.py`: assert node/edge counts AND that train/val/test
   edge splits are disjoint. Edge leakage → AUC ≈ 1.0 that is meaningless.

3. **Smoke (Phase 2)** — Cora, `--hidden_channels 512 --epochs 10`. Require a valid
   AUC in (0.5, 1.0), no NaNs.

4. **Full runs (Phase 3)** — Cora/Citeseer/Pubmed × seeds {0,1,2}, `--epochs 200`,
   `--K 20 --alpha 0.2 --init RWR`. Log AUC/AP to `results/metrics.csv`.
   **GATE: Cora mean AUC > 0.90.** If below, raise hidden dim or check `--init`.

5. **Baselines + report (Phase 4)** — compute CN / Adamic-Adar / Katz for context;
   use the `dataviz` skill for the comparison chart and a highlighted subgraph.

## VRAM discipline (8 GB)

Rarely the bottleneck here. `hidden_channels=512` is plenty for these graphs; only
raise toward 1024/8192 if AUC underperforms and VRAM allows. Use `--device cpu` to
debug logic without touching the GPU.

## Guardrails

- One conda env (`hlgnn_env`); pin PyG to the torch/CUDA build.
- Report AUC as mean ± std over 3 seeds — no seed cherry-picking.
- `data/` and `repo/` stay git-ignored.
- Run the `code-review` skill before committing custom baseline code.
