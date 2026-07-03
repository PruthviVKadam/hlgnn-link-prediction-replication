#!/usr/bin/env python
"""GATE 1 for HL-GNN: Planetoid counts + train/test edge-leakage check.

- Loads Cora / Citeseer / Pubmed via PyG (auto-downloads into ./data).
- Asserts known node/edge counts.
- Builds a link-prediction split and asserts train/val/test positive edges are
  DISJOINT. Overlap silently inflates AUC toward 1.0 — the #1 metric bug.

Usage:  python scripts/check_data.py
"""
from __future__ import annotations
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit

os.makedirs("data", exist_ok=True)

EXPECTED = {  # (nodes, undirected_edges)
    "Cora": (2708, 5429),
    "CiteSeer": (3327, 4732),
    "PubMed": (19717, 44338),
}


def edge_set(edge_index):
    e = edge_index.t().tolist()
    return {tuple(sorted(p)) for p in e}


all_ok = True
for name, (n_nodes, n_edges) in EXPECTED.items():
    ds = Planetoid("data", name)
    data = ds[0]
    got_nodes = data.num_nodes
    got_edges = data.edge_index.size(1) // 2  # undirected stored both ways
    ok_counts = got_nodes == n_nodes
    print(f"[{'PASS' if ok_counts else 'FAIL'}] {name}: {got_nodes} nodes "
          f"(expected {n_nodes}), ~{got_edges} undirected edges")
    all_ok &= ok_counts

    # Leakage check on a standard link-prediction split.
    split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                            add_negative_train_samples=False)
    train, val, test = split(data)
    s_tr = edge_set(train.edge_label_index[:, train.edge_label == 1])
    s_va = edge_set(val.edge_label_index[:, val.edge_label == 1])
    s_te = edge_set(test.edge_label_index[:, test.edge_label == 1])
    disjoint = not (s_tr & s_te) and not (s_tr & s_va) and not (s_va & s_te)
    print(f"       {'PASS' if disjoint else 'FAIL'} edge splits disjoint "
          f"(train {len(s_tr)} / val {len(s_va)} / test {len(s_te)})")
    all_ok &= disjoint

print("\nDATA CHECKS " + ("PASSED" if all_ok else "FAILED"))
raise SystemExit(0 if all_ok else 1)
