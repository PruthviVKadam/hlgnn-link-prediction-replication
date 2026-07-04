# HL-GNN — Link Prediction on Citation Graphs

Heuristic-learning GNN for link prediction.
Paper: Zhang et al., KDD 2024 · Code: [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN).

> **Result:** test AUC **96.15 / 96.50 / 98.44** on Cora / Citeseer / Pubmed —
> **+13.5 to +19.4 points** over the best classical heuristic. Full write-up with the
> baseline/paper comparison → **[REPORT.md](./REPORT.md)**.

## Quickstart

```bash
python -m venv hlgnn_env && hlgnn_env\Scripts\activate       # or conda
pip install -r requirements.txt        # torch first, then PyG — see file

python scripts/verify_env.py           # GATE 0 (torch + PyG native sparse path)
python scripts/check_data.py           # GATE 1 (counts + no edge leakage)
python scripts/check_split.py          # GATE 1b (repo split disjointness)
git clone https://github.com/LARS-research/HL-GNN.git repo
git -C repo apply ../patches/0001-*.patch    # modern-PyG + AUC-metric patch

cd repo/Planetoid
python planetoid.py --dataset cora --mlp_num_layers 3 --hidden_channels 8192 \
  --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR --runs 3

# back in project root: baselines + figures
python scripts/baselines.py            # CN / Adamic-Adar / Katz on the same split
python scripts/parse_results.py && python scripts/make_figures.py
```

> **Environment note:** compiled PyG extensions (torch-scatter/torch-sparse) have no
> Python 3.14/Windows wheels, so the upstream code is patched to PyG's native sparse
> path (`patches/`); GATE 0 verifies that path functionally.

## Documentation

| File | What it is |
|------|-----------|
| **[REPORT.md](./REPORT.md)** | Results report + **comparison vs heuristics & paper** + gate evidence |
| [PLAN.md](./PLAN.md) | Phased method with verification gates |
| [CLAUDE.md](./CLAUDE.md) | Claude Code project context |
| `.claude/` | Permission gating + `replicate-hlgnn` skill |
| `scripts/` | `verify_env.py`, `check_data.py`, `check_split.py` (gates); `baselines.py`, `make_figures.py` |

All gates 0–3 passed. Numbers: `results/metrics.csv`, `results/baselines.csv`.
