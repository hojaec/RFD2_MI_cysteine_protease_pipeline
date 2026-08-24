#!/usr/bin/env python3
"""Every round - Generate FastRelax + ProteinMPNN (MPNN-FR) tasks.

Design sequences with ProteinMPNN interleaved with Rosetta FastRelax, under the
catalytic constraints from stage 03. FIXED residues are held; alanine bias and
Met/Cys/His omission shape the rest.

Each MPNN-FR call is stochastic (temperature > 0), so ``DESIGNS_PER_BACKBONE``
independent sequences per backbone are obtained by emitting that many
independent commands per silent file (not by a single-call multiplicity flag).
Set ``DESIGNS_PER_BACKBONE`` to 15 / 40 / 25 to match round 1 / round 2 / round 3
of the published protocol.

*** Fill in the "External tools" section below before running.

Edit the CONFIG block, then:  python scripts/05_fastrelax_mpnn_taskgen.py
Submit with:                  sbatch <WORKING_DIR>/submit_mpnn_fr.sbatch
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import slurm

# ============================ CONFIG ============================
WORKING_DIR = "design/round1_mpnn_fr"
SILENT_INPUT_DIR = f"{WORKING_DIR}/silent_file"   # from stage 04
PDB_INPUT_DIR = f"{WORKING_DIR}/input_fixed"      # from stage 03
CST_FILES_DIR = f"{WORKING_DIR}/cst_files"        # from stage 03
OUTPUT_DIR = f"{WORKING_DIR}/outputs"
LOG_DIR = f"{WORKING_DIR}/log"

# Independent sequences designed per backbone: 15 (round 1), 40 (round 2), 25 (round 3).
DESIGNS_PER_BACKBONE = 15

RELAX_CYCLES = 59
TEMPERATURE = 0.1
AUGMENT_EPS = 0
BIAS_AA = '{"A": -0.5}'      # discourage over-use of alanine
OMIT_AAS = "MCH"

# --- External tools (fill in for your environment) ---
# ProteinMPNN + FastRelax driver, e.g. dl_binder_design's mpnn_fr toolkit:
#   https://github.com/nrbennet/dl_binder_design
MPNN_FR_CONTAINER = "<path-to-container>/mpnn_fr.sif"
MPNN_FR_SCRIPT = "<path-to-mpnn_fr>/dl_interface_design_cst_chAB_design_bias.py"
MPNN_FR_CHECKPOINT = "<path-to-proteinmpnn-weights>/v_48_020.pt"

# --- SLURM resources ---
TASKS_PER_JOB = 1
PARTITION = "<cpu-partition>"
MEM = "4g"
CPUS = 1
TIME = "04:00:00"
MAIL_USER = None
# ================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    commands = []
    for filename in os.listdir(SILENT_INPUT_DIR):
        if not filename.endswith(".silent"):
            continue
        base = os.path.splitext(filename)[0]
        silent_path = os.path.join(SILENT_INPUT_DIR, filename)
        pdb_path = os.path.join(PDB_INPUT_DIR, base + ".pdb")
        if not os.path.exists(pdb_path):
            print(f"Warning: no PDB for {filename}; skipping")
            continue

        for replicate in range(DESIGNS_PER_BACKBONE):
            out_dir = os.path.join(OUTPUT_DIR, f"{base}_{replicate}")
            commands.append(
                f"mkdir -p {out_dir} && cd {out_dir} && "
                f"{MPNN_FR_CONTAINER} {MPNN_FR_SCRIPT} "
                f"-silent {silent_path} "
                f"-relax_cycles {RELAX_CYCLES} "
                f"-temperature {TEMPERATURE} "
                f"-augment_eps {AUGMENT_EPS} "
                f"-fix_FIXED_res "
                f"-output_intermediates "
                f"-cst_file {CST_FILES_DIR} "
                f"-bias_AA '{BIAS_AA}' "
                f"-omit_AAs '{OMIT_AAS}' "
                f"-checkpoint_path {MPNN_FR_CHECKPOINT}"
            )

    cmds_file = os.path.join(WORKING_DIR, "cmds_mpnn_fr")
    with open(cmds_file, "w") as handle:
        handle.write("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} commands to {cmds_file} "
          f"({DESIGNS_PER_BACKBONE} per backbone)")

    slurm.write_array_script(
        os.path.join(WORKING_DIR, "submit_mpnn_fr.sbatch"),
        command_file=cmds_file,
        log_dir=LOG_DIR,
        n_commands=len(commands),
        job_name="mpnn_fr",
        partition=PARTITION,
        mem=MEM,
        cpus=CPUS,
        time=TIME,
        tasks_per_job=TASKS_PER_JOB,
        mail_user=MAIL_USER,
    )


if __name__ == "__main__":
    main()
