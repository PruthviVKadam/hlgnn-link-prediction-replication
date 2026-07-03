#!/usr/bin/env python
"""Phase 4 figures for the HL-GNN replication.

Produces (into results/):
  - auc_comparison.png   grouped bars: CN/AA/Katz/HL-GNN test AUC per dataset
  - gamma_weights.png    learned propagation weights gamma_k per dataset
  - cora_predictions.png Cora subgraph with top predicted test links highlighted

Inputs: results/baselines.csv, results/metrics.csv, results/pred_*.pt

Usage (from 2-HL-GNN root):  python scripts/make_figures.py
"""
from __future__ import annotations
import csv

import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette (dataviz reference instance, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#1baf7a", "#eda100", "#008300"  # slots 1-4

DATASETS = ["cora", "citeseer", "pubmed"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.75)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)


def fig_auc_comparison():
    base = {}  # (dataset, method) -> auc
    with open("results/baselines.csv") as f:
        for row in csv.DictReader(f):
            base[(row["dataset"], row["method"])] = float(row["test_auc"])
    hlgnn = {}
    with open("results/metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["metric"] == "AUC":
                hlgnn[row["dataset"]] = (float(row["final_test_mean"]) / 100.0,
                                         float(row["final_test_std"]) / 100.0)

    methods = [("CN", S1), ("AA", S2), ("Katz", S3), ("HL-GNN", S4)]
    x = np.arange(len(DATASETS))
    width = 0.19

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for i, (name, color) in enumerate(methods):
        if name == "HL-GNN":
            vals = [hlgnn[d][0] for d in DATASETS]
            errs = [hlgnn[d][1] for d in DATASETS]
        else:
            vals = [base[(d, name)] for d in DATASETS]
            errs = None
        pos = x + (i - 1.5) * width
        bars = ax.bar(pos, vals, width * 0.92, color=color, label=name,
                      yerr=errs, error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
        for j, (b, v) in enumerate(zip(bars, vals)):  # direct value labels
            top = v + (errs[j] if errs else 0)  # clear the error-bar cap
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, top),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK_2)

    ax.set_title("Link-prediction test AUC — heuristics vs HL-GNN "
                 "(same split, seed 234; HL-GNN mean ± std over 3 runs)",
                 color=INK, fontsize=10.5, loc="left", pad=12)
    ax.set_xticks(x, [d.capitalize() for d in DATASETS])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("test AUC", color=INK_2, fontsize=9)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncols=4,
              fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig("results/auc_comparison.png", facecolor=SURFACE,
                bbox_inches="tight")
    print("wrote results/auc_comparison.png")


def fig_gamma():
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    end_nudge = {"cora": -9, "citeseer": 9, "pubmed": 0}  # avoid label overlap
    for ds, color in zip(DATASETS, (S1, S2, S3)):
        d = torch.load(f"results/pred_{ds}.pt", weights_only=True)
        g = d["gamma"].numpy()
        ax.plot(range(len(g)), g, color=color, lw=2, marker="o", ms=4,
                label=ds.capitalize())
        ax.annotate(ds.capitalize(), (len(g) - 1, g[-1]),
                    xytext=(6, end_nudge[ds]), textcoords="offset points",
                    va="center", fontsize=8, color=color, fontweight="bold")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_title("Learned propagation weights γₖ — HL-GNN's adaptive heuristic "
                 "(hop 0 … K=20)", color=INK, fontsize=10.5, loc="left", pad=12)
    ax.set_xlabel("hop k", color=INK_2, fontsize=9)
    ax.set_ylabel("γₖ", color=INK_2, fontsize=9)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig("results/gamma_weights.png", facecolor=SURFACE,
                bbox_inches="tight")
    print("wrote results/gamma_weights.png")


def fig_cora_subgraph(top_n=20):
    d = torch.load("results/pred_cora.pt", weights_only=True)
    pos_edges = d["test_pos"].numpy()
    scores = d["pos_scores"].numpy()
    top = np.argsort(-scores)[:top_n]
    top_edges = [tuple(e) for e in pos_edges[top]]

    # context: the training graph around the highlighted nodes
    import sys
    sys.path.insert(0, "repo/Planetoid")
    from torch_geometric.datasets import Planetoid
    from utils import do_edge_split
    import os.path as osp
    dataset = Planetoid(osp.join("data", "cora"), "cora")
    split = do_edge_split(dataset)
    train_edges = split["train"]["edge"].numpy()

    keep = set()
    for u, v in top_edges:
        keep.add(u), keep.add(v)
    g = nx.Graph()
    for u, v in train_edges:
        if u in keep or v in keep:
            g.add_edge(int(u), int(v))
    g.add_edges_from(top_edges)

    pos = nx.spring_layout(g, seed=42, k=0.35)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRID, width=0.7)
    nx.draw_networkx_edges(g, pos, edgelist=top_edges, ax=ax,
                           edge_color=S1, width=2.2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=22, node_color=MUTED,
                           linewidths=0)
    hi_nodes = sorted(keep)
    nx.draw_networkx_nodes(g, pos, nodelist=hi_nodes, ax=ax, node_size=42,
                           node_color=S1, linewidths=1.2, edgecolors=SURFACE)
    ax.set_title(f"Cora — top {top_n} predicted held-out links (blue) in their "
                 "training-graph neighborhood (gray)",
                 color=INK, fontsize=10.5, loc="left", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/cora_predictions.png", facecolor=SURFACE,
                bbox_inches="tight")
    print("wrote results/cora_predictions.png")


if __name__ == "__main__":
    fig_auc_comparison()
    fig_gamma()
    fig_cora_subgraph()
