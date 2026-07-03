#!/usr/bin/env python
"""GATE 1b for HL-GNN: leakage check on the REPO'S OWN edge split.

check_data.py validates PyG's RandomLinkSplit; this validates the split the
training code actually uses (repo/Planetoid/utils.py::do_edge_split, seeded
234). Asserts train/valid/test positive edges are pairwise disjoint as
undirected pairs, and that test negatives don't collide with any positives.

Usage (from 2-HL-GNN root):  python scripts/check_split.py [cora|citeseer|pubmed]
"""
from __future__ import annotations
import os.path as osp
import sys

sys.path.insert(0, "repo/Planetoid")  # import the (patched) upstream utils
from torch_geometric.datasets import Planetoid  # noqa: E402
from utils import do_edge_split  # noqa: E402


def pairs(t):
    """Edge tensor [E, 2] -> set of undirected node pairs."""
    return {tuple(sorted(e)) for e in t.tolist()}


def main() -> int:
    names = sys.argv[1:] or ["cora", "citeseer", "pubmed"]
    all_ok = True
    for name in names:
        dataset = Planetoid(osp.join("data", name), name)
        split = do_edge_split(dataset)

        tr = pairs(split["train"]["edge"])
        va = pairs(split["valid"]["edge"])
        te = pairs(split["test"]["edge"])
        va_n = pairs(split["valid"]["edge_neg"])
        te_n = pairs(split["test"]["edge_neg"])

        pos_disjoint = not (tr & va) and not (tr & te) and not (va & te)
        pos_all = tr | va | te
        neg_clean = not (te_n & pos_all) and not (va_n & pos_all)

        ok = pos_disjoint and neg_clean
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
              f"train {len(tr)} / valid {len(va)} / test {len(te)} positives; "
              f"pos_disjoint={pos_disjoint}, negatives_clean={neg_clean}")
    print("\nREPO SPLIT CHECKS " + ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
