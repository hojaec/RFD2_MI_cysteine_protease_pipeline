#!/usr/bin/env python3
"""Round 0, step 2 - Filter RFdiffusion backbones by secondary structure.

Annotate each diffusion output's chain-A secondary structure (DSSP-style H/S/L)
and keep only backbones that are not dominated by loops. Passing PDBs (and
their ``.trb`` metadata) are copied to a ``filtered_output`` folder, which
becomes the backbone pool for round 1.

*** Fill in UTIL_PATHS below: this depends on RFdiffusion's own pdb_util /
*** calc_dssp helper modules and PyTorch, both available in the RFdiffusion
*** environment, not part of this repo.

Edit the CONFIG block, then:  python scripts/02_filter_by_dssp.py
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
import sys

# ============================ CONFIG ============================
WORKING_DIR = "diffusion/round0_scaffold"
OUTPUT_SUBDIR = "filtered_output"          # created under WORKING_DIR
SSE_CSV = "backbone_sse.csv"               # written under WORKING_DIR
FILTERED_CSV = "backbone_sse_filtered.csv"

# Filters: reject a backbone if it has this many consecutive loop residues ...
MAX_CONSECUTIVE_LOOP = 7
# ... or if the loop fraction of chain A exceeds this.
MAX_LOOP_FRACTION = 0.3

# External tools: directories containing RFdiffusion's pdb_util / calc_dssp.
UTIL_PATHS = [
    "<path-to-rfdiffusion-aa>",
    "<path-to-your-local-util-dir>",
]
# ================================================================


def _load_rfdiffusion_helpers():
    """Import RFdiffusion's pdb_util/calc_dssp (available in that environment)."""
    for path in UTIL_PATHS:
        if path not in sys.path:
            sys.path.append(path)
    from pdb_util import parse_pdb
    from calc_dssp import annotate_sse
    return parse_pdb, annotate_sse


def has_consecutive(sequence, char, count):
    return char * count in sequence


def annotate_all(all_pdbs, csv_path):
    parse_pdb, annotate_sse = _load_rfdiffusion_helpers()
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "H", "S", "L", "X", "file_path", "dssp_sequence"])
        for pdb in all_pdbs:
            parse = parse_pdb(pdb)
            chain_a = [i for i, ch in enumerate(parse["pdb_idx"]) if ch[0] == "A"]
            if not chain_a:
                continue
            ca_xyz = parse["xyz"][chain_a, 1, :]          # CA of chain A
            dssp = annotate_sse(ca_xyz)
            fraction = dssp.sum(dim=0) / dssp.shape[0]
            seq = "".join(
                "HSLX"[idx] for idx in dssp.argmax(dim=1).tolist()
            )
            writer.writerow(
                [os.path.basename(pdb), fraction[0].item(), fraction[1].item(),
                 fraction[2].item(), fraction[3].item(), pdb, seq]
            )


def filter_and_copy(sse_csv, filtered_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    kept = []
    with open(sse_csv) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            if has_consecutive(row["dssp_sequence"], "L", MAX_CONSECUTIVE_LOOP):
                continue
            if float(row["L"]) > MAX_LOOP_FRACTION:
                continue
            kept.append(row)
            pdb_path = row["file_path"]
            shutil.copy(pdb_path, output_dir)
            trb_path = os.path.join(
                os.path.dirname(pdb_path),
                os.path.basename(pdb_path).replace(".pdb", ".trb"),
            )
            if os.path.exists(trb_path):
                shutil.copy(trb_path, output_dir)

    with open(filtered_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def main():
    sse_csv = os.path.join(WORKING_DIR, SSE_CSV)
    filtered_csv = os.path.join(WORKING_DIR, FILTERED_CSV)
    output_dir = os.path.join(WORKING_DIR, OUTPUT_SUBDIR)

    all_pdbs = glob.glob(os.path.join(WORKING_DIR, "outputs", "*", "*.pdb"))
    print(f"Annotating {len(all_pdbs)} backbones ...")
    annotate_all(all_pdbs, sse_csv)

    n_kept = filter_and_copy(sse_csv, filtered_csv, output_dir)
    print(f"Kept {n_kept} backbones -> {output_dir}")
    print(f"Filtered table: {filtered_csv}")


if __name__ == "__main__":
    main()
