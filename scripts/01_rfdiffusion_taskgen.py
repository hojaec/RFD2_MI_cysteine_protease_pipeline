#!/usr/bin/env python3
"""Round 0, step 1 - Generate RFdiffusion motif-scaffolding tasks.

Scaffolds a protein backbone around each catalytic motif (from stage 00) while
docking it against the substrate hotspot, using an all-atom RFdiffusion model.
Writes one command per (motif PDB x contig permutation) and a SLURM array
script. Total backbones produced = NUM_DESIGNS x number of (motif, permutation)
pairs; set NUM_DESIGNS so that total lands at the target backbone count for
this round (50,000 in the published protocol).

*** Fill in the placeholders in the "External tools" section below before
*** running: this repo ships no model weights, containers, or cluster
*** configuration.

Edit the CONFIG block, then:  python scripts/01_rfdiffusion_taskgen.py
Submit with:                  sbatch <WORKING_DIR>/submit_rfdiffusion.sbatch
"""

from __future__ import annotations

import glob
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import slurm

# ============================ CONFIG ============================
WORKING_DIR = "diffusion/round0_scaffold"
INPUT_DIR = "motifs/combined"              # motif PDBs from stage 00
OUTPUT_DIR = f"{WORKING_DIR}/outputs"
LOG_DIR = f"{WORKING_DIR}/log"
PROJECT_NAME = "round0_scaffold"

# Backbones generated per (motif, contig-permutation) command. Total output
# count = NUM_DESIGNS * len(motif PDBs) * len(permutations of PERMUTATION_GROUPS).
NUM_DESIGNS = 50_000
LENGTH_RANGE = "180-200"           # contigmap.length
HOTSPOT_RES = "B95"                # substrate hotspot residue
WRITE_EXTRA_TS = "[30,20,10]"
CUSTOM_T_RANGE = "[50,49,48,47,46,45,44,43,42,41,40,38,36,34,32,30,27,23,20,17,14,10,7,4,1]"

# Motif segments whose order is permuted to diversify topology.
PERMUTATION_GROUPS = ["A102-131", "A173-185"]
# Contig template around the permuted segments; {p0}/{p1} are filled per permutation.
CONTIG_TEMPLATE = "30-100,{p0},10-25,{p1},30-100_2,B94-99"

# --- External tools (fill in for your environment) ---
# All-atom RFdiffusion inference entrypoint, e.g. an RFdiffusion-AA checkout:
#   https://github.com/RosettaCommons/RFdiffusion
RFDIFFUSION_SCRIPT = "<path-to-rfdiffusion-aa>/run_inference.py --config-name=aa_ppi"
# Model checkpoint (public weights, or your own fine-tune).
RFDIFFUSION_CHECKPOINT = "<path-to-checkpoint>/RFD_all_atom.pt"

# --- SLURM resources (edit partition/resource names for your cluster) ---
TASKS_PER_JOB = 1
PARTITION = "<gpu-partition>"
GRES = "<gpu-resource>"            # e.g. "gpu:1"
MEM = "16g"
CPUS = 2
TIME = "6:00:00"
MAIL_USER = None                   # e.g. "yourusername"; None omits the directive
# ================================================================


def main():
    for path in (INPUT_DIR, OUTPUT_DIR, LOG_DIR):
        os.makedirs(path, exist_ok=True)

    cmd_template = (
        f"{RFDIFFUSION_SCRIPT} inference.ckpt_path={RFDIFFUSION_CHECKPOINT} "
        f"inference.write_extra_ts={WRITE_EXTRA_TS} "
        f"inference.custom_t_range={CUSTOM_T_RANGE} "
        f"transforms.configs.CenterPostTransform.center_type=target_hotspot "
        f"inference.num_designs={NUM_DESIGNS} contigmap.length={LENGTH_RANGE} "
        f"ppi.hotspot_res='\"{HOTSPOT_RES}\"' "
    )

    permutations = list(itertools.permutations(PERMUTATION_GROUPS))
    pdb_files = glob.glob(os.path.join(INPUT_DIR, "*.pdb"))
    if not pdb_files:
        raise SystemExit(f"No motif PDBs in {INPUT_DIR}")

    commands = []
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file).replace(".pdb", "")
        for i, perm in enumerate(permutations):
            contigs = CONTIG_TEMPLATE.format(p0=perm[0], p1=perm[1])
            output_prefix = f"{OUTPUT_DIR}/{name}_contig_{i}/{name}_contig_{i}"
            commands.append(
                f"{cmd_template} "
                f"inference.input_pdb={pdb_file} "
                f"inference.output_prefix={output_prefix} "
                f"contigmap.contigs=[\"'{contigs}'\"]"
            )

    task_file = os.path.join(WORKING_DIR, "rfdiffusion_task.sh")
    with open(task_file, "w") as handle:
        handle.write("\n".join(commands) + "\n")
    total_backbones = NUM_DESIGNS * len(commands)
    print(f"Wrote {len(commands)} commands to {task_file}")
    print(f"Expected total backbones: {total_backbones}")

    slurm.write_array_script(
        os.path.join(WORKING_DIR, "submit_rfdiffusion.sbatch"),
        command_file=task_file,
        log_dir=LOG_DIR,
        n_commands=len(commands),
        job_name=PROJECT_NAME,
        partition=PARTITION,
        gres=GRES,
        mem=MEM,
        cpus=CPUS,
        time=TIME,
        tasks_per_job=TASKS_PER_JOB,
        mail_user=MAIL_USER,
    )


if __name__ == "__main__":
    main()
