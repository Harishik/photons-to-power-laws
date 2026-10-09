#!/usr/bin/env python3
"""Regenerate the synthetic demo outputs and README figures.

Equivalent to ``p2pl demo --out examples/synthetic --figures docs/figures --animate``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from grb_afterglow.cli import main

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    args = ["demo", "--out", str(ROOT / "examples" / "synthetic"), "--figures", str(ROOT / "docs" / "figures")]
    sys.exit(main([*args, "--animate", *sys.argv[1:]]))
