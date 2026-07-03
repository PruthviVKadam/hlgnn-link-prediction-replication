#!/usr/bin/env python
"""Parse the Phase 3 training logs into results/metrics.csv.

Reads results/train_{cora,citeseer,pubmed}.txt, extracts the end-of-run
"All runs:" statistics (mean ± std over --runs) for AUC and Hits@K, and
writes one tidy CSV row per (dataset, metric).

Usage (from 2-HL-GNN root):  python scripts/parse_results.py
"""
from __future__ import annotations
import csv
import re

DATASETS = ["cora", "citeseer", "pubmed"]
METRICS = ["AUC", "Hits@10", "Hits@50", "Hits@100"]

# tolerant of the ± glyph being mangled by console encoding
FINAL = re.compile(r"Final Test:\s*([\d.]+)\s*.\s*([\d.]+|nan)")
VALID = re.compile(r"Highest Valid:\s*([\d.]+)\s*.\s*([\d.]+|nan)")

rows = []
for ds in DATASETS:
    lines = open(f"results/train_{ds}.txt", encoding="utf-8",
                 errors="replace").read().splitlines()
    # the final summary is the LAST "<metric>\nAll runs:" header; take the
    # 4 statistics lines that follow it
    for metric in METRICS:
        starts = [i for i, ln in enumerate(lines)
                  if ln.strip() == metric
                  and i + 1 < len(lines) and lines[i + 1].strip() == "All runs:"]
        if not starts:
            print(f"[WARN] {ds}: no 'All runs' block for {metric}")
            continue
        block = "\n".join(lines[starts[-1] + 2: starts[-1] + 6])
        m_final, m_valid = FINAL.search(block), VALID.search(block)
        if not m_final:
            print(f"[WARN] {ds}/{metric}: no Final Test line")
            continue
        rows.append({
            "dataset": ds,
            "metric": metric,
            "final_test_mean": float(m_final.group(1)),
            "final_test_std": (float(m_final.group(2))
                               if m_final.group(2) != "nan" else 0.0),
            "highest_valid_mean": float(m_valid.group(1)) if m_valid else "",
        })
        print(f"{ds:10s} {metric:9s} final test = "
              f"{m_final.group(1)} ± {m_final.group(2)}")

with open("results/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "metric", "final_test_mean",
                                      "final_test_std", "highest_valid_mean"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote results/metrics.csv ({len(rows)} rows)")
