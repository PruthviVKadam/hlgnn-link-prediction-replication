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

## Results (fill with real numbers, mean ± std over 3 seeds)

| Dataset  | AUC | AP | Notes |
|----------|-----|----|-------|
| Cora     | _TBD (bar > 0.90)_ | | |
| Citeseer | _TBD_ | | |
| Pubmed   | _TBD_ | | |

## Files

- `PLAN.md` — phased plan with verification gates
- `CLAUDE.md` — Claude Code project context
- `.claude/` — permission gating + `replicate-hlgnn` skill
- `scripts/verify_env.py` (torch+PyG gate), `scripts/check_data.py` (counts + leakage)
