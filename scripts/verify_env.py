#!/usr/bin/env python
"""GATE 0 preflight for the HL-GNN replication.

The critical check is that torch AND torch_geometric (plus its C++ extensions
torch_scatter / torch_sparse) import together against the SAME CUDA build. A
mismatch is the #1 failure mode and shows up as `undefined symbol` or a segfault.

Usage:  python scripts/verify_env.py
"""
from __future__ import annotations
import shutil
import sys

MIN_PY = (3, 9)
MIN_FREE_DISK_GB = 1.0
REQUIRE_CUDA = False   # graphs are tiny; CPU is acceptable for debugging

checks: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def main() -> int:
    v = sys.version_info
    record("Python >= %d.%d" % MIN_PY, v[:2] >= MIN_PY, "found %d.%d.%d" % v[:3])

    try:
        import torch
        record("PyTorch import", True, torch.__version__)
        cuda = torch.cuda.is_available()
        record("CUDA available", cuda or not REQUIRE_CUDA, "cuda=%s" % cuda)
        if cuda:
            record("GPU visible", True, torch.cuda.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        record("PyTorch import", False, repr(e))

    # The make-or-break import: PyG + its compiled extensions.
    try:
        import torch_geometric
        record("torch_geometric import", True, torch_geometric.__version__)
    except Exception as e:  # noqa: BLE001
        record("torch_geometric import", False, repr(e))
    for ext in ("torch_scatter", "torch_sparse"):
        try:
            __import__(ext)
            record(f"{ext} import", True, "ok")
        except Exception as e:  # noqa: BLE001
            record(f"{ext} import", False, repr(e))

    free_disk_gb = shutil.disk_usage(".").free / 1e9
    record("Free disk >= %.1f GB" % MIN_FREE_DISK_GB,
           free_disk_gb >= MIN_FREE_DISK_GB, "%.1f GB free" % free_disk_gb)

    width = max(len(n) for n, _, _ in checks)
    print("\nHL-GNN environment preflight\n" + "-" * (width + 20))
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print("-" * (width + 20))
    if all_ok:
        print("ALL CHECKS PASSED\n")
        return 0
    print("PREFLIGHT FAILED — a torch/PyG version mismatch is the usual cause.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
