#!/usr/bin/env python3
"""Every round - Label catalytic residues FIXED and write Rosetta constraints.

Run this on the backbone pool entering FastRelax + MPNN in any round (the DSSP-
filtered round-0 backbones, or a round's partial-diffusion output). For each
backbone this:
  * finds the Cys-His-Asp triad in chain A,
  * tags those residues with a ``FIXED`` PDBInfo label (so ProteinMPNN + FR
    leaves them alone),
  * writes a Rosetta constraint file wiring the triad + oxyanion hole + scissile
    peptide into a productive geometry (see protease_pipeline/constraints.py).

Requires PyRosetta (https://www.pyrosetta.org, licence required).

Edit the CONFIG block, then:  python scripts/03_fix_motif_and_constraints.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import constraints

# ============================ CONFIG ============================
# Point INPUT_DIR at this round's backbone pool, e.g.:
#   round 1: diffusion/round0_scaffold/filtered_output
#   round 2: diffusion/round2_partial_T40/outputs/<merged backbones>
#   round 3: diffusion/round3_partial_T10/outputs/<merged backbones>
INPUT_DIR = "diffusion/round0_scaffold/filtered_output"
WORKING_DIR = "design/round1_mpnn_fr"
OUTPUT_DIR = f"{WORKING_DIR}/input_fixed"        # FIXED-labelled PDBs
CST_DIR = f"{WORKING_DIR}/cst_files"

# Substrate register: scissile P1 carbonyl carbon and P1' leaving-group N sit at
# these offsets into chain B (pose index = chain_a_length + offset).
P1_CARBONYL_OFFSET = 8
LEAVING_GROUP_OFFSET = 9
# ================================================================


def main():
    from pyrosetta import init, pose_from_pdb
    from pyrosetta.rosetta import core

    init(extra_options="--mute all")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CST_DIR, exist_ok=True)

    input_pdbs = glob.glob(os.path.join(INPUT_DIR, "*.pdb"))
    n_ok = 0
    for pdb_path in input_pdbs:
        pdb_file = os.path.basename(pdb_path)
        pose = pose_from_pdb(pdb_path)
        pdb_info = core.pose.PDBInfo(pose)

        his = asp = cys = None
        fixed_residues = []
        for i in range(1, pose.total_residue() + 1):
            if pose.chain(i) != 1:  # chain A only
                continue
            name = pose.residue(i).name3()
            if name == "HIS":
                his = i
                fixed_residues.append(i)
            elif name == "ASP":
                asp = i
                fixed_residues.append(i)
            elif name == "CYS":
                cys = i
                fixed_residues.append(i)

        if not (his and asp and cys):
            print(f"Skipping {pdb_file}: missing one of His/Asp/Cys in chain A")
            continue

        for i in fixed_residues:
            pdb_info.add_reslabel(i, "FIXED")
        pose.pdb_info(pdb_info)
        pose.dump_pdb(os.path.join(OUTPUT_DIR, pdb_file))

        cst_text = constraints.catalytic_triad_constraints(
            his=his,
            asp=asp,
            cys=cys,
            chain_a_length=pose.chain_end(1),
            p1_carbonyl_offset=P1_CARBONYL_OFFSET,
            leaving_group_offset=LEAVING_GROUP_OFFSET,
        )
        with open(os.path.join(CST_DIR, pdb_file.replace(".pdb", ".cst")), "w") as handle:
            handle.write(cst_text)
        n_ok += 1

    print(f"Labelled + constrained {n_ok}/{len(input_pdbs)} PDBs")
    print(f"  FIXED PDBs -> {OUTPUT_DIR}")
    print(f"  constraints -> {CST_DIR}")


if __name__ == "__main__":
    main()
