# HL-GNN Replication Plan

> **Paper:** Zhang et al., *Heuristic Learning with Graph Neural Networks: A Unified
> Framework for Link Prediction*, KDD 2024.
> **Code:** [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN) (PyTorch + PyG)
> **Goal here:** Reproduce link-prediction results on the Planetoid citation graphs
> (Cora, Citeseer, Pubmed) on the target laptop (RTX 4070, 8 GB VRAM).

## Objective & success criteria

- **Primary metric:** link-prediction **AUC** (and Average Precision) on Cora / Citeseer / Pubmed.
- **Success bar:** HL-GNN AUC > 0.90 on Cora (small citation graphs are "easy"; a plain
  Adamic-Adar heuristic already reaches ~0.80, so HL-GNN should clearly beat it).
- **Report:** mean ± std over ≥ 3 seeds per dataset.

## Hardware budget (must stay under)

| Resource | Budget | Lever if exceeded |
|----------|--------|-------------------|
| VRAM | ≤ 8 GB (aim 2–4 GB) | lower `--hidden_channels` (8192 → 512), fewer layers |
| RAM | ≤ 32 GB (aim ~2 GB) | graphs are tiny |
| Disk | < 100 MB | PyG auto-downloads Planetoid |
| Time | minutes per dataset | reduce `--epochs` for smoke test |

**Note:** the repo's example uses `hidden_channels=8192`. That is unnecessary on
8 GB VRAM for these small graphs — start at **512** and only raise if AUC underperforms.

---

## Phased plan (each phase ends with a GATE)

### Phase 0 — Environment (version-sensitive!)

PyG must match the installed torch build. This is the highest-risk step.

```bash
conda create -n hlgnn_env python=3.9 -y && conda activate hlgnn_env
# torch first, matching CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# then PyG wheels built for THAT torch/CUDA:
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
```

- **GATE 0:** `python scripts/verify_env.py` prints `ALL CHECKS PASSED` — confirms
  torch **and** torch_geometric import together and (optionally) CUDA is visible.
  Version mismatch between torch and PyG is the #1 setup failure — this gate catches it.

### Phase 1 — Code & data

```bash
git clone https://github.com/LARS-research/HL-GNN.git repo
# PyG downloads Cora/Citeseer/Pubmed on first run into ./data
```

- **GATE 1:** `python scripts/check_data.py` loads each Planetoid graph and asserts
  node/edge counts (Cora 2708/5429, Citeseer 3327/4732, Pubmed 19717/44338) and that
  train/val/test **edge splits do not overlap** (leakage check — see Pitfalls).

### Phase 2 — Smoke test (Cora, few epochs)

```bash
cd repo/Planetoid
python planetoid.py --dataset cora --mlp_num_layers 3 --hidden_channels 512 \
  --dropout 0.5 --epochs 10 --K 20 --alpha 0.2 --init RWR
```

- **GATE 2:** runs without CUDA/PyG errors, prints a valid AUC (0.5–1.0), no NaNs.

### Phase 3 — Full runs (all three datasets, multi-seed)

```bash
for ds in cora citeseer pubmed; do
  for seed in 0 1 2; do
    python planetoid.py --dataset $ds --hidden_channels 512 \
      --epochs 200 --K 20 --alpha 0.2 --init RWR --seed $seed
  done
done
```

- Capture AUC/AP per run to `results/metrics.csv`.
- **GATE 3 (success bar):** Cora mean AUC > 0.90. If below, raise `hidden_channels`
  (512 → 1024) or check the `--init` (RWR = global heuristic init).

### Phase 4 — Baselines & report

- Compute heuristic baselines (Common Neighbors, Adamic-Adar, Katz) for context.
- **`dataviz` skill:** bar chart HL-GNN vs heuristics per dataset; a subgraph with
  top predicted links highlighted.
- **`verify` skill:** drive one full train→eval cycle end-to-end before declaring done.

---

## Pitfalls (from the report + PyG gotchas)

- **torch ↔ PyG version mismatch** — pin PyG wheels to the exact torch/CUDA build
  (GATE 0). Symptom: `undefined symbol` / segfault on `import torch_geometric`.
- **Edge leakage** — the same edge appearing in train and test inflates AUC to ~1.0.
  GATE 1 asserts split disjointness; align with the repo's transductive split.
- **CUDA OOM** at high `hidden_channels` — start at 512; use `--device cpu` to debug.
- **`--K` too high** — more hops = more compute; 20 is the paper default, don't inflate.
- Fix `torch.manual_seed` for the ±std to be meaningful.

## Extensions (optional, for showcase)

- Heuristic baselines (CN / AA / Katz) vs HL-GNN AUC table.
- Feature ablation (`--use_node_feat False`) — topology-only performance.
- Depth study (1 vs 3 vs 10 MLP layers) — oversmoothing curve.
- Local vs global init (`init=CN` vs `init=RWR`) — the unified-framework benefit.
- Stretch: one OGB dataset (`ogbl-collab`) — may need CPU for memory.

## Deployment (optional)

Gradio app: user supplies a small edge list, model returns missing-link probabilities.
Dockerfile from a PyTorch base + PyG wheels.

## Definition of done — ✅ COMPLETED 2026-07-03

- [x] Gates 0–3 passed (evidence in [REPORT.md](./REPORT.md); env strategy
      changed to native-sparse PyG — no compiled extensions, see REPORT § 5)
- [x] Cora/Citeseer/Pubmed AUC (mean ± std, 3 runs) in `results/metrics.csv`
- [x] Cora AUC > 0.90 recorded — **96.15 ± 0.93**
- [x] Baseline-vs-HL-GNN chart generated (`results/auc_comparison.png`)
- [x] `README.md` results table filled with real numbers
