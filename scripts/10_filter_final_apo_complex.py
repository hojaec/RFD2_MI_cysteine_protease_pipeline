#!/usr/bin/env python3
"""Round 3 - Final filter: joint apo + complex AF3 confidence and agreement.

Round 3 runs AF3 twice per design (stage 06/07 with MODE="apo" and
MODE="complex"). A design is accepted only if:

  complex  complex_plddt > COMPLEX_PLDDT_MIN, pae_interface < PAE_INTERFACE_MAX
           (AF3's minimum inter-chain PAE), iptm > IPTM_MIN, and
           key_plddt > KEY_PLDDT_MIN (catalytic triad + substrate contact).
  apo      plddt > APO_PLDDT_MIN, and the apo (unbound) chain-A model agrees
           with chain A of the complex model to within RMSD_APO_TO_COMPLEX_MAX
           CA-RMSD - i.e. binding the substrate does not require an induced
           conformational change the enzyme cannot access on its own.

Each mode's AF3 run can yield several diffusion samples per design. For each
description this script keeps the best complex sample that passes the complex
thresholds and the best apo sample that passes the apo pLDDT threshold, then
checks apo/complex agreement between exactly that pair.

Edit the CONFIG block, then:  python scripts/10_filter_final_apo_complex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import geometry

# ============================ CONFIG ============================
APO_CSV = "af3/round3_apo/af3_data.csv"
COMPLEX_CSV = "af3/round3_complex/af3_data.csv"
OUTPUT_CSV = "af3/round3_final/af3_data_filtered.csv"

# Complex thresholds.
COMPLEX_PLDDT_MIN = 90
PAE_INTERFACE_MAX = 1.2
IPTM_MIN = 0.9
KEY_PLDDT_MIN = 85

# Apo thresholds.
APO_PLDDT_MIN = 90
RMSD_APO_TO_COMPLEX_MAX = 0.8
# ================================================================


def _best_row(df, sort_col):
    """Row with the highest ``sort_col`` value, or None if df is empty."""
    if df.empty:
        return None
    return df.loc[df[sort_col].idxmax()]


def main():
    apo = pd.read_csv(APO_CSV)
    complex_ = pd.read_csv(COMPLEX_CSV)

    for col in ("plddt", "ptm", "rmsd_to_mpnn"):
        if col in apo.columns:
            apo[col] = pd.to_numeric(apo[col], errors="coerce")
    for col in ("complex_plddt", "pae_interface", "iptm", "key_plddt"):
        if col in complex_.columns:
            complex_[col] = pd.to_numeric(complex_[col], errors="coerce")

    complex_pass = complex_[
        (complex_["complex_plddt"] > COMPLEX_PLDDT_MIN)
        & (complex_["pae_interface"] < PAE_INTERFACE_MAX)
        & (complex_["iptm"] > IPTM_MIN)
        & (complex_["key_plddt"] > KEY_PLDDT_MIN)
    ]
    apo_pass = apo[apo["plddt"] > APO_PLDDT_MIN]

    accepted = []
    for description in sorted(set(complex_pass["description"]) & set(apo_pass["description"])):
        best_complex = _best_row(complex_pass[complex_pass["description"] == description], "complex_plddt")
        best_apo = _best_row(apo_pass[apo_pass["description"] == description], "plddt")
        if best_complex is None or best_apo is None:
            continue

        rmsd_apo_to_complex = geometry.calculate_ca_rmsd(
            best_apo["pdb_path"], best_complex["pdb_path"], chain1="A", chain2="A"
        )
        if rmsd_apo_to_complex is None or rmsd_apo_to_complex >= RMSD_APO_TO_COMPLEX_MAX:
            continue

        row = best_complex.to_dict()
        row.update({
            "apo_pdb_path": best_apo["pdb_path"],
            "apo_plddt": best_apo["plddt"],
            "rmsd_apo_to_complex": rmsd_apo_to_complex,
        })
        accepted.append(row)

    result = pd.DataFrame(accepted)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Designs passing complex thresholds: {complex_pass['description'].nunique()}")
    print(f"Designs passing apo pLDDT: {apo_pass['description'].nunique()}")
    print(f"Designs passing the joint apo/complex filter: {len(result)}")
    print(f"Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
