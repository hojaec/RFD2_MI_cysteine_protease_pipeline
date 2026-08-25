#!/usr/bin/env python3
"""Setup - Build catalytic-triad motif inputs.

One-time setup step, run before round 0. Enumerates combinations of His / Asp /
Cys catalytic rotamers together with the target substrate and writes one merged
motif PDB per combination. These motif PDBs are the input geometry that
RFdiffusion (01) scaffolds a protein around.

Edit the CONFIG block, then:  python scripts/00_build_catalytic_motifs.py
"""

from __future__ import annotations

import glob
import itertools
import os

# ============================ CONFIG ============================
# Directory holding the individual rotamer/target PDBs.
INPUT_DIR = "motifs/rotamers"
# Where the merged motif PDBs are written.
OUTPUT_DIR = "motifs/combined"

# Glob patterns (relative to INPUT_DIR) for each motif component.
HIS_GLOB = "his_*.pdb"
ASP_GLOB = "asp_*.pdb"
CYS_GLOB = "cys_*.pdb"
TARGET_GLOB = "target_*.pdb"

# Output file naming: motif_<index>.pdb, numbering starting here.
OUTPUT_PREFIX = "motif_"
START_INDEX = 1
# ================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    his_files = sorted(glob.glob(os.path.join(INPUT_DIR, HIS_GLOB)))
    asp_files = sorted(glob.glob(os.path.join(INPUT_DIR, ASP_GLOB)))
    cys_files = sorted(glob.glob(os.path.join(INPUT_DIR, CYS_GLOB)))
    target_files = sorted(glob.glob(os.path.join(INPUT_DIR, TARGET_GLOB)))

    if not target_files:
        raise SystemExit(f"No target PDB found with pattern {TARGET_GLOB} in {INPUT_DIR}")
    for label, files, pattern in [
        ("His", his_files, HIS_GLOB),
        ("Asp", asp_files, ASP_GLOB),
        ("Cys", cys_files, CYS_GLOB),
    ]:
        if not files:
            raise SystemExit(f"No {label} PDB found with pattern {pattern} in {INPUT_DIR}")

    combinations = itertools.product(his_files, asp_files, cys_files, target_files)
    count = 0
    for i, (his_file, asp_file, cys_file, target_file) in enumerate(
        combinations, start=START_INDEX
    ):
        output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{i}.pdb")
        with open(output_path, "w") as outfile:
            for component in (his_file, asp_file, cys_file, target_file):
                with open(component) as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
        count += 1

    print(f"Wrote {count} motif PDBs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
