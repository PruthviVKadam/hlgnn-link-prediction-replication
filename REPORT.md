# HL-GNN Link Prediction — Replication Report

**Paper:** Zhang et al., *Heuristic Learning with Graph Neural Networks: A Unified
Framework for Link Prediction*, KDD 2024.
**Code:** [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN) (PyTorch + PyG)
**Completed:** 2026-07-03 · all gates 0–3 passed · RTX 4070 Laptop (8 GB)

Results report. To run it see [README.md](./README.md); for the phased method and
gates see [PLAN.md](./PLAN.md).

---

## 1. What was replicated

HL-GNN learns a *generalized graph heuristic* for link prediction: it propagates node
features up to K=20 hops and learns a per-hop weight γₖ, so it can express Common
Neighbors, Katz, PageRank, etc. as special cases and interpolate between them. We
reproduced link prediction on the three **Planetoid citation graphs** (Cora, Citeseer,
Pubmed), reporting **ROC-AUC** (primary) and Hits@K, mean ± std over 3 runs.

## 2. Setup

| Item | Value |
|------|-------|
| Datasets | Cora (2708 nodes), Citeseer (3327), Pubmed (19717) |
| Split | repo's `do_edge_split` (seed 234, 85/5/10) — **same split reused for baselines** |
| Hyperparameters | official README configs (Cora/Cite: hidden 8192, RWR init; Pubmed: 512, KI) |
| Runs | 3 per dataset (model re-init; the transductive split is fixed by design) |
| Env | Python 3.14.3, torch 2.11.0+cu128, **PyG 2.8.0 native sparse path** (no compiled ext) |
| Cost | ~15 min for all 9 runs; 2–4 GB VRAM |

## 3. Results

Final test metrics, mean ± std over 3 runs:

| Dataset | **AUC** | Hits@100 | Hits@50 | Hits@10 |
|---------|---------|----------|---------|---------|
| Cora | **96.15 ± 0.93** | 94.69 ± 2.14 | 88.68 ± 1.23 | 75.59 ± 2.70 |
| Citeseer | **96.50 ± 0.73** | 96.04 ± 1.52 | 90.84 ± 2.79 | 76.19 ± 1.89 |
| Pubmed | **98.44 ± 0.06** | 88.09 ± 0.57 | 78.57 ± 0.85 | 54.35 ± 3.85 |

Figures in [`results/`](./results/): `auc_comparison.png` (below),
`gamma_weights.png` (learned per-hop weights), `cora_predictions.png` (top predicted
held-out links on a subgraph).

## 4. Comparison — our output vs baselines and paper

### 4a. vs classical heuristics (computed by us on the identical split)

This is the paper's central claim — a *learned* heuristic beats fixed ones. We
computed Common Neighbors, Adamic-Adar, and Katz on the exact same seed-234 split, so
these deltas are apples-to-apples. See `results/auc_comparison.png`.

| Dataset | CN | Adamic-Adar | Katz (best fixed) | **HL-GNN (ours)** | Δ vs Katz |
|---------|-----|-------------|-------------------|-------------------|-----------|
| Cora | 0.7228 | 0.7242 | 0.8262 | **0.9615** | **+13.5** |
| Citeseer | 0.6770 | 0.6773 | 0.7713 | **0.9650** | **+19.4** |
| Pubmed | 0.6384 | 0.6385 | 0.7923 | **0.9844** | **+19.2** |

HL-GNN beats the strongest fixed heuristic by **+13.5 to +19.4 AUC points** — the
paper's headline result, reproduced.

### 4b. vs the paper's reported numbers

The paper reports Hits@100 on Planetoid. Our Hits@100 (94.7 / 96.0 / 88.1) sits in the
**same range the paper reports** (~94–96 on Cora/Citeseer, ~88 on Pubmed) — i.e. we
reproduce the paper's operating point, not just "beats baselines." Exact paper table
values are not transcribed here to avoid citing numbers we did not regenerate; the
defensible, self-computed comparison is 4a.

### 4c. why the learned heuristic wins — the γₖ weights

`results/gamma_weights.png` shows the learned per-hop weights decaying with distance
but going **slightly negative** at k ≳ 14 on Cora/Citeseer. No fixed heuristic
(RWR/KI, which are non-negative geometric series) can express a sign change — this is
a concrete visual of *why* the learned heuristic outperforms them.

## 5. Changes made to the upstream repo

The plan's #1 flagged risk (torch↔PyG matching) materialized as: **compiled PyG
extensions have no Python 3.14/Windows wheels.** Rather than downgrade, we ported the
code to PyG's native sparse path. Diff in
[`patches/0001-modern-pyg-auc-and-paths.patch`](./patches/0001-modern-pyg-auc-and-paths.patch).

1. **`Planetoid/model.py`** — `torch_sparse.matmul` → `torch_geometric.utils.spmm`;
   dead `torch_scatter`/`SparseTensor` imports removed; unpack `gcn_norm`'s tuple
   return on PyG ≥ 2.8. *(The rewritten GATE 0 functional check caught this tuple
   change before any training time was spent — the version-gate earning its keep.)*
2. **`Planetoid/utils.py`** — dead imports removed; `DataLoader` import relocation
   fixed (PyG ≥ 2.4); `train_test_split_edges` vendored as a fallback.
3. **`Planetoid/planetoid.py`** — **AUC added as the primary metric** (the repo only
   computed Hits@K via the OGB evaluator); `edge_weight` guard; in-project data path;
   `--save_pred` to export test scores + learned γₖ.

## 6. Verification gates (evidence)

- **GATE 0** — torch 2.11.0+cu128 + PyG 2.8.0; native sparse path
  (ToSparseTensor→gcn_norm→spmm) functionally verified. First run *failed* on the
  PyG 2.8 tuple return — fixed and re-passed.
- **GATE 1a/1b** — Planetoid node counts (2708/3327/19717); repo's own `do_edge_split`
  has pairwise-disjoint train/valid/test positives and clean negatives on all three
  datasets (`scripts/check_split.py`). Edge leakage → AUC≈1.0 is the classic silent
  bug; this gate rules it out.
- **GATE 2** — Cora smoke (10 ep): test AUC 96.47, no NaN.
- **GATE 3** — Cora AUC 96.15 ≫ 0.90 bar; full grid above.

## 7. Honest caveats

- CN/AA baseline AUCs (~0.64–0.72) look low vs older literature because they are
  computed on the 90%-train graph only and suffer massive score ties at 0; Katz
  (global) is the fair strong heuristic, and it's the one HL-GNN is measured against.
- Baselines and model share one fixed split; the ±std reflects model re-init only.
- scipy fancy indexing rejects torch's non-writeable `.numpy()` views (the Katz scorer
  needed an explicit `.copy()`).
