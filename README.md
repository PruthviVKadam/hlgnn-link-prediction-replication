# HL-GNN — Link Prediction on Citation Graphs

Heuristic-learning GNN for link prediction.
Paper: Zhang et al., KDD 2024 · Code: [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN).

## Quickstart

```bash
conda create -n hlgnn_env python=3.9 -y && conda activate hlgnn_env
# torch first (match CUDA), then PyG wheels pinned to that build — see requirements.txt
pip install -r requirements.txt

python scripts/verify_env.py          # GATE 0 (torch + PyG import together)
python scripts/check_data.py          # GATE 1 (counts + no edge leakage)
git clone https://github.com/LARS-research/HL-GNN.git repo

cd repo/Planetoid
python planetoid.py --dataset cora --hidden_channels 512 --epochs 200 \
  --K 20 --alpha 0.2 --init RWR
```

Full method, gates, and pitfalls: [PLAN.md](./PLAN.md).

## Results — replication completed 2026-07-03

Final test metrics, mean ± std over 3 runs (identical `do_edge_split` seed-234
split for model and baselines):

| Dataset  | **HL-GNN AUC**   | Hits@100     | Best heuristic (Katz) | Margin |
|----------|------------------|--------------|-----------------------|--------|
| Cora     | **96.15 ± 0.93** | 94.69 ± 2.14 | 82.62                 | +13.5  |
| Citeseer | **96.50 ± 0.73** | 96.04 ± 1.52 | 77.13                 | +19.4  |
| Pubmed   | **98.44 ± 0.06** | 88.09 ± 0.57 | 79.23                 | +19.2  |

✅ GATE 3 (Cora AUC > 0.90) passed with 6-point margin. Hits@100 consistent
with the paper's reported Planetoid numbers. Total training: ~15 min for all
9 runs on the RTX 4070 (8 GB), official README hyperparameters.

Figures in [`results/`](./results/): `auc_comparison.png` (heuristics vs
HL-GNN), `gamma_weights.png` (learned γₖ — note the negative long-range
weights on Cora/Citeseer), `cora_predictions.png` (top predicted held-out
links). Full numbers: `results/metrics.csv`, `results/baselines.csv`.
Deviations & patches: [NOTES.md](./NOTES.md).

> **Environment note:** the compiled PyG extensions the plan called for have
> no Python 3.14/Windows wheels — the upstream code is instead patched to
> PyG's native sparse path (see `patches/`), and GATE 0 verifies that path
> functionally.

## Files

- `PLAN.md` — phased plan with verification gates
- `CLAUDE.md` — Claude Code project context
- `.claude/` — permission gating + `replicate-hlgnn` skill
- `scripts/verify_env.py` (torch+PyG gate), `scripts/check_data.py` (counts + leakage)
