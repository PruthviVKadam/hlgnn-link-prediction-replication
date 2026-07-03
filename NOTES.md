# Replication notes — changes & deviations log

Changes made to facilitate the Planetoid replication. Everything else follows
[PLAN.md](./PLAN.md) verbatim.

## Environment deviations

- **No conda** — plan assumed a conda env; conda isn't installed, so a plain
  venv at `C:\Users\pruth\venvs\hlgnn_env` (outside OneDrive, same rationale as
  the SimCLR project) with Python 3.14.3 + torch 2.11.0+cu128 + PyG 2.8.0.
- **No compiled PyG extensions** — the plan's Phase 0 prescribed
  `pyg-lib/torch-scatter/torch-sparse` wheels pinned to the torch build. Those
  wheels don't exist for Python 3.14 on Windows, so the strategy changed:
  patch the upstream code to PyG's **native sparse path** instead (below).
  `scripts/verify_env.py` was rewritten to functionally verify that path
  (ToSparseTensor → gcn_norm → spmm on a toy graph) rather than requiring the
  extension imports.

## Patches to `repo/` (LARS-research/HL-GNN)

> Exported to [`patches/0001-modern-pyg-auc-and-paths.patch`](./patches/0001-modern-pyg-auc-and-paths.patch).
> Recreate: `git clone https://github.com/LARS-research/HL-GNN.git repo && git -C repo apply ../patches/0001-*.patch`

1. **`Planetoid/model.py`** — `torch_sparse.matmul` → `torch_geometric.utils.spmm`;
   unused `torch_scatter`/`SparseTensor` imports removed; unpack `gcn_norm`'s
   tuple return `(adj, edge_weight)` on PyG ≥ 2.8 (the GATE 0 functional check
   caught this before training).
2. **`Planetoid/utils.py`** — three dead `torch_sparse` imports removed;
   `DataLoader` import fixed (moved out of `torch_geometric.data` in PyG ≥ 2.4);
   `train_test_split_edges` vendored as a fallback (still present in PyG 2.8,
   deprecated — insurance for future versions).
3. **`Planetoid/planetoid.py`** — **AUC added as the primary metric** (the plan
   specifies AUC; the repo only computed Hits@K via the OGB evaluator) with an
   `AUC` logger reported alongside Hits@{10,50,100}; `edge_weight` guard for
   modern PyG `Data`; data root moved from `~/dataset` to the in-project
   `data/` (shared with the gate scripts); new `--save_pred` flag exports test
   scores + the learned γₖ weights for visualization.

## Run configuration

Official README configs (not the plan's conservative `hidden_channels 512` —
VRAM allows the paper values; plan permits raising):

| Dataset  | mlp layers | hidden | dropout | epochs | init | runs |
|----------|-----------|--------|---------|--------|------|------|
| cora     | 3         | 8192   | 0.5     | 100    | RWR  | 3    |
| citeseer | 2         | 8192   | 0.5     | 100    | RWR  | 3    |
| pubmed   | 3         | 512    | 0.6     | 300    | KI   | 3    |

Split: repo's `do_edge_split` (seed 234, 85/5/10) — identical split reused for
the heuristic baselines, so the comparison is apples-to-apples. Multi-run
variation comes from model re-initialization (`--runs 3`), the transductive
split itself is fixed by design.

## Gate evidence

- **GATE 0**: all preflight checks passed; native sparse path functionally
  verified. (First run FAILED on the PyG 2.8 `gcn_norm` tuple return — exactly
  the class of version bug the functional gate exists to catch; fixed in
  model.py and re-passed.)
- **GATE 1a**: Planetoid node counts verified (2708/3327/19717); PyG
  RandomLinkSplit disjoint.
- **GATE 1b**: repo's own `do_edge_split` split — train/valid/test positives
  pairwise disjoint, negatives collide with no positives, all three datasets
  (`scripts/check_split.py`).
- **GATE 2**: Cora smoke (10 epochs): test AUC 96.47, Hits@100 96.02, no NaN.
- **GATE 3**: full runs, final test (mean ± std over 3 runs):

| Dataset  | **AUC** | Hits@100 | Hits@50 | Hits@10 | wall time |
|----------|---------|----------|---------|---------|-----------|
| cora     | **96.15 ± 0.93** | 94.69 ± 2.14 | 88.68 ± 1.23 | 75.59 ± 2.70 | ~3.5 min |
| citeseer | **96.50 ± 0.73** | 96.04 ± 1.52 | 90.84 ± 2.79 | 76.19 ± 1.89 | ~3.3 min |
| pubmed   | **98.44 ± 0.06** | 88.09 ± 0.57 | 78.57 ± 0.85 | 54.35 ± 3.85 | ~8.2 min |

  Cora AUC 96.15 ≫ 0.90 bar. Hits@100 values are consistent with the paper's
  reported Planetoid results (~94–96 Cora/Citeseer, ~88 Pubmed).

## Baselines (same split)

| Dataset  | CN | AA | Katz | HL-GNN |
|----------|-----|-----|------|--------|
| cora     | 0.7228 | 0.7242 | 0.8262 | **0.9615** |
| citeseer | 0.6770 | 0.6773 | 0.7713 | **0.9650** |
| pubmed   | 0.6384 | 0.6385 | 0.7923 | **0.9844** |

+13 to +19 AUC points over the strongest heuristic (Katz) — consistent with
the paper's headline claim.

## Post-run observations

- The learned γₖ weights (`results/gamma_weights.png`) decay with hop distance
  but go **slightly negative** at k ≳ 14 on Cora/Citeseer — adaptive behavior
  no fixed heuristic (RWR/KI) can express; a nice visual of the paper's
  "learned generalized heuristic" claim.
- CN/AA baseline AUCs (~0.64–0.72) look low vs older literature because they
  are computed on the 90%-train graph only and suffer massive score ties at 0;
  Katz (global) is the fair strong heuristic here.
- scipy fancy indexing rejects torch's non-writeable `.numpy()` views
  (`baselines.py` Katz scorer needed explicit `.copy()`).
