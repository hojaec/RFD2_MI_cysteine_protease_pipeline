#!/usr/bin/env python3
"""Every round - Pack FIXED PDBs into Rosetta silent files.

FastRelax + ProteinMPNN (05) reads Rosetta silent files. This writes a bash
script (one ``silentfrompdbsparallel`` call per PDB) that converts the FIXED-
labelled PDBs from stage 03 into per-backbone silent files.

*** Fill in SILENT_TOOL below: this is part of the dl_binder_design toolkit
*** (https://github.com/nrbennet/dl_binder_design, ``include/silent_tools``),
*** not bundled here.

Edit the CONFIG block, then:  python scripts/04_pdbs_to_silent.py
Run the generated script:     bash <WORKING_DIR>/cmds_make_silent_file
"""

from __future__ import annotations

import glob
import os

# ============================ CONFIG ============================
WORKING_DIR = "design/round1_mpnn_fr"
PDB_INPUT = f"{WORKING_DIR}/input_fixed"      # FIXED PDBs from stage 03
SILENT_OUTPUT = f"{WORKING_DIR}/silent_file"

# External tool: silent_tools converter from dl_binder_design.
SILENT_TOOL = "<path-to-dl_binder_design>/include/silent_tools/silentfrompdbsparallel"
# ================================================================


def main():
    os.makedirs(SILENT_OUTPUT, exist_ok=True)
    pdb_files = glob.glob(os.path.join(PDB_INPUT, "*.pdb"))
    if not pdb_files:
        raise SystemExit(f"No PDB files in {PDB_INPUT}")

    commands = []
    for pdb_file in pdb_files:
        base = os.path.basename(pdb_file)
        silent_file = os.path.join(SILENT_OUTPUT, base.replace(".pdb", ".silent"))
        commands.append(
            f'find {PDB_INPUT} -name "{base}" | xargs -n 1 {SILENT_TOOL} >> {silent_file}'
        )

    commands_file = os.path.join(WORKING_DIR, "cmds_make_silent_file")
    with open(commands_file, "w") as handle:
        handle.write("#!/bin/bash\n\n")
        handle.write("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} commands to {commands_file}")


if __name__ == "__main__":
    main()
