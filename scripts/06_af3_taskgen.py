#!/usr/bin/env python3
"""Every round - Generate AlphaFold3 prediction tasks (apo or complex).

Reads chain sequences from each FastRelax+MPNN design model and writes one
single-seed AF3 JSON per design (MSAs/templates disabled for speed), the run
commands, and a SLURM array script.

Set ``MODE``:
  * "apo"     - chain A (the designed protease) alone. Used in every round to
                check the enzyme folds correctly on its own.
  * "complex" - chain A + chain B (protease + substrate). Used in round 3
                alongside "apo" to additionally check the bound geometry.

Round 3 needs both: run this script twice (MODE="apo" then MODE="complex"),
each into its own WORKING_DIR.

*** Fill in the "External tools" section below before running.

Edit the CONFIG block, then:  python scripts/06_af3_taskgen.py
Submit with:                  sbatch <WORKING_DIR>/submit_af3.sh
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import pdb_utils, slurm

# ============================ CONFIG ============================
MODE = "apo"                       # "apo" | "complex"

WORKING_DIR = "af3/round1_apo"
INPUT_FOLDER = "design/round1_mpnn_fr/pdbs"  # FastRelax+MPNN design PDBs (chains A + B)
JSON_DIR = f"{WORKING_DIR}/json_files"
OUTPUT_DIR = f"{WORKING_DIR}/output"
LOG_DIR = f"{WORKING_DIR}/log"

MODEL_SEEDS = [1]

# --- External tools (fill in for your environment) ---
# AlphaFold3: https://github.com/google-deepmind/alphafold3
AF3_CONTAINER = "<path-to-container>/alphafold3.sif"
AF3_ENTRY = "python <path-to-alphafold3>/run_alphafold_custom.py"

# --- SLURM resources ---
GROUP_SIZE = 60                # commands per array task
PARTITION = "<gpu-partition>"
GRES = "<gpu-resource>"
MEM = "8G"
CPUS = 2
TIME = "01:30:00"
MAIL_USER = None
# ================================================================


def _af3_json(name, chains):
    sequences = [
        {"protein": {"id": chain_id, "sequence": seq, "unpairedMsa": "",
                     "pairedMsa": "", "templates": ""}}
        for chain_id, seq in chains.items()
    ]
    return [{
        "name": name,
        "sequences": sequences,
        "modelSeeds": MODEL_SEEDS,
        "dialect": "alphafold3",
        "version": 1,
    }]


def main():
    if MODE not in ("apo", "complex"):
        raise SystemExit(f"MODE must be 'apo' or 'complex', got {MODE!r}")
    required_chains = ("A",) if MODE == "apo" else ("A", "B")

    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd_file = os.path.join(WORKING_DIR, "cmds_af3")

    n = 0
    with open(cmd_file, "w") as handle:
        for file_name in os.listdir(INPUT_FOLDER):
            if not file_name.lower().endswith(".pdb"):
                continue
            name = os.path.splitext(file_name)[0]
            seqs = pdb_utils.extract_chain_sequences(
                os.path.join(INPUT_FOLDER, file_name), chains=("A", "B")
            )
            if any(c not in seqs for c in required_chains):
                print(f"Missing required chain(s) for {name}; skipping")
                continue

            chains = {c: seqs[c] for c in required_chains}
            json_path = os.path.join(JSON_DIR, f"{name}.json")
            with open(json_path, "w") as jf:
                json.dump(_af3_json(name, chains), jf, separators=(",", ":"))

            handle.write(
                f"{AF3_CONTAINER} {AF3_ENTRY} "
                f"--json_path={json_path} --output_dir={OUTPUT_DIR}\n"
            )
            n += 1

    print(f"Wrote {n} AF3 ({MODE}) commands to {cmd_file}")
    slurm.write_array_script(
        os.path.join(WORKING_DIR, "submit_af3.sh"),
        command_file=cmd_file,
        log_dir=LOG_DIR,
        n_commands=n,
        job_name=f"af3_{MODE}",
        partition=PARTITION,
        gres=GRES,
        mem=MEM,
        cpus=CPUS,
        time=TIME,
        tasks_per_job=GROUP_SIZE,
        mail_user=MAIL_USER,
    )


if __name__ == "__main__":
    main()
