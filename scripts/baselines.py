#!/usr/bin/env python
"""Heuristic link-prediction baselines on the repo's exact edge split.

Computes AUC for Common Neighbors (CN), Adamic-Adar (AA), and truncated Katz
on the same do_edge_split(seed=234) split the HL-GNN training uses, so the
numbers in results/baselines.csv are directly comparable to the model runs.

Usage (from 2-HL-GNN root):  python scripts/baselines.py
"""
from __future__ import annotations
import csv
import os.path as osp
import sys

import numpy as np
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "repo/Planetoid")  # patched upstream utils
from torch_geometric.datasets import Planetoid  # noqa: E402
from utils import do_edge_split  # noqa: E402

KATZ_BETA = 0.05
KATZ_HOPS = 3


def build_adj(train_edges, num_nodes) -> ssp.csr_matrix:
    r = train_edges[:, 0].numpy()
    c = train_edges[:, 1].numpy()
    # train edges from do_edge_split are already stored in both directions
    a = ssp.csr_matrix((np.ones(len(r)), (r, c)), shape=(num_nodes, num_nodes))
    a.data[:] = 1.0  # dedupe any doubled entries
    return a


def cn_scores(a, edges):
    src, dst = edges[:, 0].numpy(), edges[:, 1].numpy()
    return np.asarray(a[src].multiply(a[dst]).sum(axis=1)).flatten()


def aa_scores(a, edges):
    deg = np.asarray(a.sum(axis=0)).flatten()
    w = 1.0 / np.log(np.maximum(deg, 2.0))  # guard deg<2 -> log>=log 2
    src, dst = edges[:, 0].numpy(), edges[:, 1].numpy()
    common = a[src].multiply(a[dst])  # rows: indicator of common neighbors
    return common @ w


def katz_matrix(a) -> ssp.csr_matrix:
    s = KATZ_BETA * a
    term = a
    for hop in range(2, KATZ_HOPS + 1):
        term = term @ a
        term.data[:] = np.minimum(term.data, 1e6)  # numerical guard
        s = s + (KATZ_BETA ** hop) * term
    return s.tocsr()


def katz_scores(k, edges):
    src, dst = edges[:, 0].numpy(), edges[:, 1].numpy()
    return np.asarray(k[src, dst]).flatten()


def auc(pos, neg):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return roc_auc_score(y, np.concatenate([pos, neg]))


def main() -> int:
    rows = []
    for name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(osp.join("data", name), name)
        num_nodes = dataset[0].num_nodes
        split = do_edge_split(dataset)
        a = build_adj(split["train"]["edge"], num_nodes)
        pos, neg = split["test"]["edge"], split["test"]["edge_neg"]

        scores = {
            "CN": (cn_scores(a, pos), cn_scores(a, neg)),
            "AA": (aa_scores(a, pos), aa_scores(a, neg)),
            "Katz": (katz_scores(katz_matrix(a), pos),
                     katz_scores(katz_matrix(a), neg)),
        }
        for method, (sp, sn) in scores.items():
            val = auc(sp, sn)
            rows.append({"dataset": name, "method": method,
                         "test_auc": round(val, 4)})
            print(f"{name:10s} {method:5s} test AUC = {val:.4f}")

    with open("results/baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "test_auc"])
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
