#!/usr/bin/env python3
"""Rounds 1 & 2 - Filter designs by apo AF3 confidence + fold accuracy.

Applied after each round's apo (chain-A-only) AF3 prediction (stages 06/07 with
MODE="apo"). Requires the predicted apo structure to closely match the intended
fold (low CA-RMSD to the FastRelax+MPNN design model) and to be confidently
predicted (pLDDT, pTM).

Published thresholds:
  round 1: RMSD_TO_MPNN_MAX=1.8, PLDDT_MIN=80, PTM_MIN=0.8
  round 2: RMSD_TO_MPNN_MAX=1.2, PLDDT_MIN=85, PTM_MIN=0.85

Edit the CONFIG block, then:  python scripts/08_filter_apo_round.py
"""

from __future__ import annotations

import pandas as pd

# ============================ CONFIG ============================
INPUT_CSV = "af3/round1_apo/af3_data.csv"
OUTPUT_CSV = "af3/round1_apo/af3_data_filtered.csv"

RMSD_TO_MPNN_MAX = 1.8
PLDDT_MIN = 80
PTM_MIN = 0.8
# ================================================================


def main():
    df = pd.read_csv(INPUT_CSV)
    for col in ("plddt", "ptm", "rmsd_to_mpnn"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cond = (
        (df["rmsd_to_mpnn"] < RMSD_TO_MPNN_MAX)
        & (df["plddt"] > PLDDT_MIN)
        & (df["ptm"] > PTM_MIN)
    )
    passing = df[cond]
    passing.to_csv(OUTPUT_CSV, index=False)
    print(f"Passing samples: {len(passing)} / {len(df)}")
    print(f"Unique designs passing: {passing['description'].nunique()}")
    print(f"Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
