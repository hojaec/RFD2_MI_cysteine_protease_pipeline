#!/usr/bin/env python3
"""Rounds 2 & 3 - Generate partial-diffusion refinement tasks.

Diversify each round's accepted backbones with partial RFdiffusion
(``diffuser.partial_T``): re-noise a design to the given depth and re-denoise to
explore nearby topologies while keeping the docked motif. The contig for each
design is reconstructed from its ``.trb`` ``sampled_mask`` and renumbered to
match the design PDB; the substrate hotspot is placed relative to chain B's
start residue. ``NUM_DESIGNS_PER_BACKBONE`` independent output backbones are
produced per input backbone in a single RFdiffusion call.

Published settings:
  round 2: PARTIAL_T=40, NUM_DESIGNS_PER_BACKBONE=15
  round 3: PARTIAL_T=10, NUM_DESIGNS_PER_BACKBONE=30

*** Fill in the "External tools" section below before running.

Edit the CONFIG block, then:  python scripts/09_partial_diffusion_taskgen.py
Submit with:                  sbatch <WORKING_DIR>/submit_partial_diffusion.sbatch
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protease_pipeline import pdb_utils, slurm

# ============================ CONFIG ============================
PROJECT_NAME = "round2_partial_T40"
WORKING_DIR = f"diffusion/{PROJECT_NAME}"
# Backbones accepted by the previous round's filter (their .trb files must sit
# alongside the PDBs, or be reachable via TRB_DIR below).
INPUT_DIR = f"{WORKING_DIR}/partial_input"
TRB_DIR = f"{WORKING_DIR}/partial_input"
OUTPUT_DIR = f"{WORKING_DIR}/outputs"
LOG_DIR = f"{WORKING_DIR}/log"

PARTIAL_T = 40
NUM_DESIGNS_PER_BACKBONE = 15
HOTSPOT_OFFSET = 6            # hotspot = chain B start residue + this
OUTPUT_TAG = "partial_T40"    # subfolder / prefix tag on each output

# --- External tools (fill in for your environment) ---
RFDIFFUSION_SCRIPT = "<path-to-rfdiffusion-aa>/run_inference.py --config-name=aa_ppi"
RFDIFFUSION_CHECKPOINT = "<path-to-checkpoint>/RFD_all_atom.pt"

# --- SLURM resources ---
TASKS_PER_JOB = 1
PARTITION = "<gpu-partition>"
GRES = "<gpu-resource>"
MEM = "16g"
CPUS = 2
TIME = "12:00:00"
MAIL_USER = None
# ================================================================


def renumber_chain_a(chain_a_str):
    """Renumber chain-A segments of a sampled_mask so numbering is contiguous.

    ``sampled_mask`` alternates gap tokens ("63-63") and A-segment tokens
    ("A43-56"). Each A segment is re-based to start just after the preceding gap.
    """
    tokens = [t.strip() for t in chain_a_str.split(",")]
    new_tokens = []
    last_a_new_end = None
    gap_before = None
    for token in tokens:
        if not token.startswith("A"):
            new_tokens.append(token)
            gap_before = int(token.split("-")[0])
        else:
            orig_start, orig_end = map(int, token[1:].split("-"))
            seg_length = orig_end - orig_start
            if gap_before is None:
                raise ValueError("Expected a gap token before an A segment")
            if last_a_new_end is None:
                new_start = gap_before + 1
            else:
                new_start = last_a_new_end + gap_before + 1
            new_end = new_start + seg_length
            new_tokens.append(f"A{new_start}-{new_end}")
            last_a_new_end = new_end
    return ",".join(new_tokens)


def renumber_chain_b(chain_b_str, start_num_b):
    """Offset chain-B segment numbers by the substrate's start residue."""
    tokens = [t.strip() for t in chain_b_str.split(",")]
    new_tokens = []
    for token in tokens:
        if token.startswith("B"):
            orig_start, orig_end = map(int, token[1:].split("-"))
            new_tokens.append(f"B{start_num_b + orig_start}-{start_num_b + orig_end}")
        else:
            new_tokens.append(token)
    return ",".join(new_tokens)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    cmd_template = (
        f"{RFDIFFUSION_SCRIPT} inference.ckpt_path={RFDIFFUSION_CHECKPOINT} "
        f"inference.num_designs={NUM_DESIGNS_PER_BACKBONE} diffuser.partial_T={PARTIAL_T} "
        f"upstream_inference_transforms.configs.RenumberCroppedInput.enabled=False "
        f"diffuser.independently_center_diffuseds=False "
    )

    commands = []
    for pdb_file in glob.glob(os.path.join(INPUT_DIR, "*.pdb")):
        name = os.path.basename(pdb_file).replace(".pdb", "")
        trb_path = os.path.join(TRB_DIR, f"{name}.trb")
        if not os.path.exists(trb_path):
            print(f"TRB not found for {name}; skipping")
            continue
        start_num_b = pdb_utils.get_chain_start_residue(pdb_file, chain="B")
        if start_num_b is None:
            print(f"Chain B not found in {pdb_file}; skipping")
            continue

        with open(trb_path, "rb") as handle:
            sampled_mask = pickle.load(handle).get("sampled_mask", [])
        if len(sampled_mask) < 2:
            raise ValueError(f"sampled_mask needs two parts in {trb_path}")

        new_a = renumber_chain_a(sampled_mask[0])
        new_b = renumber_chain_b(sampled_mask[1], start_num_b)
        contigmap = f'"\'{new_a}_{new_b}\'"'
        hotspot = f"B{start_num_b + HOTSPOT_OFFSET}"
        output_prefix = f"{OUTPUT_DIR}/{name}_{OUTPUT_TAG}/{name}_{OUTPUT_TAG}"

        commands.append(
            f"{cmd_template} "
            f"ppi.hotspot_res='\"{hotspot}\"' "
            f"inference.input_pdb={pdb_file} "
            f"inference.output_prefix={output_prefix} "
            f"contigmap.contigs=[{contigmap}]"
        )

    task_file = os.path.join(WORKING_DIR, "partial_diffusion_task.sh")
    with open(task_file, "w") as handle:
        handle.write("\n".join(commands) + "\n")
    total_backbones = NUM_DESIGNS_PER_BACKBONE * len(commands)
    print(f"Wrote {len(commands)} commands to {task_file}")
    print(f"Expected total backbones: {total_backbones} "
          f"({NUM_DESIGNS_PER_BACKBONE} per input backbone)")

    slurm.write_array_script(
        os.path.join(WORKING_DIR, "submit_partial_diffusion.sbatch"),
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
