#!/usr/bin/env python3
"""Every round - Collect AlphaFold3 confidences into a CSV (apo or complex).

Walk the AF3 output tree (one folder per design, each with
``seed-1_sample-*`` subfolders) written by stage 06 and record per sample:

  MODE = "apo"      plddt, ptm, and rmsd_to_mpnn (CA-RMSD of the predicted
                     chain-A-only model against chain A of the FastRelax+MPNN
                     design model that produced this sequence).
  MODE = "complex"   complex_plddt, pae_interface (mean of the A<->B chain-pair
                     PAE minima - AF3's "min PAE"), iptm, and key_plddt (mean
                     pLDDT over the catalytic triad + the substrate's scissile
                     backbone, residues 8-9 of chain B).

Also renames each ``model.cif`` to a descriptive name.

Edit the CONFIG block, then:  python scripts/07_collect_af3.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import geometry, pdb_utils

# ============================ CONFIG ============================
MODE = "apo"                       # "apo" | "complex" - must match stage 06

WORKING_DIR = "af3/round1_apo"
OUTPUT_ACT_DIR = f"{WORKING_DIR}/output"      # top-level folder of AF3 results
COMBINED_OUTPUT_CSV = f"{WORKING_DIR}/af3_data.csv"

# MODE == "apo": design model directory, to compute rmsd_to_mpnn against chain A.
DESIGN_MODEL_DIR = "design/round1_mpnn_fr/pdbs"

# MODE == "complex": template PDB directory, to look up the catalytic triad's
# residue numbers (chain A) for key_plddt.
TEMPLATE_PDB_DIR = "design/round1_mpnn_fr/input_fixed"

# Substrate scissile residues (chain B) whose backbone counts toward key_plddt
# (MODE == "complex" only).
SUBSTRATE_KEY_RESIDUES = {8, 9}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
# ================================================================


def _design_model_for(description):
    """Design PDB matching a description, stripping the FastRelax/MPNN suffix."""
    stem = description.split("_dldesign_", 1)[0]
    return os.path.join(DESIGN_MODEL_DIR, f"{stem}.pdb")


def _template_for(description):
    stem = description.split("_dldesign_", 1)[0]
    return os.path.join(TEMPLATE_PDB_DIR, f"{stem}.pdb")


def _key_plddt(cif_path, atom_plddts, chainA_key_residues):
    atoms = pdb_utils.parse_mmcif_atoms(cif_path)
    if not atoms:
        return None
    if len(atoms) != len(atom_plddts):
        print(f"Warning: atom count {len(atoms)} != plddt count {len(atom_plddts)} ({cif_path})")
    scores = []
    for i, (chain, resseq, _resname, atom_name) in enumerate(atoms):
        if i >= len(atom_plddts):
            break
        if chain == "A" and resseq in chainA_key_residues:
            scores.append(atom_plddts[i])
        elif chain == "B" and resseq in SUBSTRATE_KEY_RESIDUES and atom_name in BACKBONE_ATOMS:
            scores.append(atom_plddts[i])
    return sum(scores) / len(scores) if scores else None


def main():
    if MODE not in ("apo", "complex"):
        raise SystemExit(f"MODE must be 'apo' or 'complex', got {MODE!r}")

    rows = []
    for description in os.listdir(OUTPUT_ACT_DIR):
        desc_dir = os.path.join(OUTPUT_ACT_DIR, description)
        if not os.path.isdir(desc_dir):
            continue

        chainA_key = None
        if MODE == "complex":
            chainA_key = pdb_utils.key_residues_from_chainA(_template_for(description))
            if not chainA_key:
                print(f"Warning: no key residues in template for {description}")

        for sample_dir in glob.glob(os.path.join(desc_dir, "seed-1_sample-*")):
            base = os.path.basename(sample_dir)
            m = re.search(r"seed-1_sample-(\d+)", base)
            if not m:
                continue
            sample_num = m.group(1)

            conf_file = os.path.join(sample_dir, "confidences.json")
            summary_file = os.path.join(sample_dir, "summary_confidences.json")
            if not (os.path.exists(conf_file) and os.path.exists(summary_file)):
                print(f"Missing confidence files in {sample_dir}")
                continue

            with open(conf_file) as f:
                atom_plddts = json.load(f).get("atom_plddts", [])
            if not atom_plddts:
                continue
            avg_plddt = sum(atom_plddts) / len(atom_plddts)

            with open(summary_file) as f:
                summary = json.load(f)

            original_cif = os.path.join(sample_dir, "model.cif")
            named_cif = os.path.join(
                sample_dir, f"{description}_seed-1_sample-{sample_num}_model.cif"
            )
            if not os.path.exists(named_cif):
                if os.path.exists(original_cif):
                    os.rename(original_cif, named_cif)
                else:
                    print(f"No model.cif in {sample_dir}")
                    continue

            pdb_path = os.path.join(
                desc_dir, base, f"{description}_seed-1_sample-{sample_num}_model.pdb"
            )

            if MODE == "apo":
                ptm = summary.get("ptm")
                if ptm is None:
                    print(f"'ptm' not found in {summary_file}")
                    continue
                design_model = _design_model_for(description)
                rmsd_to_mpnn = None
                if os.path.exists(design_model):
                    rmsd_to_mpnn = geometry.calculate_ca_rmsd(
                        pdb_path, design_model, chain1="A", chain2="A"
                    )
                else:
                    print(f"Design model not found: {design_model}")
                rows.append({
                    "description": description,
                    "pdb_path": pdb_path,
                    "plddt": avg_plddt,
                    "ptm": ptm,
                    "rmsd_to_mpnn": rmsd_to_mpnn,
                })
            else:
                chain_pair = summary.get("chain_pair_pae_min")
                if not chain_pair or len(chain_pair) < 2 or len(chain_pair[0]) < 2:
                    continue
                pae_interface = (chain_pair[0][1] + chain_pair[1][0]) / 2.0
                iptm = summary.get("iptm")
                if iptm is None:
                    continue
                rows.append({
                    "description": description,
                    "pdb_path": pdb_path,
                    "complex_plddt": avg_plddt,
                    "pae_interface": pae_interface,
                    "iptm": iptm,
                    "key_plddt": _key_plddt(named_cif, atom_plddts, chainA_key),
                })

    pd.DataFrame(rows).to_csv(COMBINED_OUTPUT_CSV, index=False)
    print(f"Collected {len(rows)} samples ({MODE}) -> {COMBINED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
