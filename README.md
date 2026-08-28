# De novo cysteine protease design pipeline

Computational pipeline for the **de novo design of cysteine proteases** built
around a **Cys–His–Asp catalytic triad** and an oxyanion hole, docked against a
peptide substrate. Backbones are generated with all-atom RFdiffusion, sequences
are designed with ProteinMPNN under Rosetta FastRelax and catalytic geometric
constraints, and candidates are filtered over three rounds of increasingly
strict AlphaFold3 confidence and structural-agreement criteria.

This repository contains the analysis/orchestration code: the code that builds
task lists for the heavy models, parses their outputs, and applies the
filters. **It does not bundle any model weights, containers, or cluster
configuration** — every script has a clearly marked `CONFIG` block, and every
external tool path is a placeholder (`<...>`) that you must fill in for your
own environment before running anything. See
[Environment setup](#environment-setup).

**Want to see the code run before installing any of that?**
[`demo/`](demo/README.md) has a small simulated dataset and walks through the
pure-Python stages (motif assembly, AF3-confidence collection, the round
filters) end-to-end with no external tools required.

---

## Design concept

Serine/cysteine proteases cleave a peptide bond using a catalytic triad. Here the
nucleophile is a **cysteine**, activated by a **histidine** general base, which
is in turn oriented by an **aspartate**. The scissile carbonyl of the bound
substrate is stabilised in an **oxyanion hole** formed by backbone amides. A
design is only accepted if AlphaFold3 reproduces this arrangement with the
correct distances/angles and does so confidently and reproducibly — geometry
and agreement between independent predictions, not just fold confidence, are
the filter.

The two-chain convention used throughout:

| Chain | Contents                                              |
|-------|-------------------------------------------------------|
| `A`   | Designed protease (carries the Cys–His–Asp triad)     |
| `B`   | Bound peptide substrate (scissile bond at P1–P1′)     |

"Apo" predictions fold chain A alone (no substrate); "complex" predictions fold
chain A + chain B together.

---

## Pipeline overview

Three rounds, each narrowing the pool and tightening the acceptance bar. Every
round follows the same generate → design → predict → filter shape; only the
backbone-generation step and the thresholds change.

```
 [00] build catalytic motif  (His/Asp/Cys rotamers + substrate)
        │
 [01] RFdiffusion  ──▶  50,000 backbones
        │
 [02] DSSP filter   (reject loop-dominated backbones)
        │
════════ ROUND 1 ════════════════════════════════════════════════
 [03] fix triad + Rosetta constraints
 [04] pack silent files
 [05] FastRelax + ProteinMPNN   ──▶  15 sequences / backbone
 [06,07] AlphaFold3, apo (chain A only)
 [08] filter: RMSD-to-design < 1.8 Å, pLDDT > 80, pTM > 0.8
════════ ROUND 2 ════════════════════════════════════════════════
 [09] partial diffusion (T=40)  ──▶  15 backbones / accepted design
 [03-05] fix + constraints + silent + FastRelax/MPNN ──▶ 40 sequences / backbone
 [06,07] AlphaFold3, apo
 [08] filter: RMSD-to-design < 1.2 Å, pLDDT > 85, pTM > 0.85
════════ ROUND 3 ════════════════════════════════════════════════
 [09] partial diffusion (T=10)  ──▶  30 backbones / accepted design
 [03-05] fix + constraints + silent + FastRelax/MPNN ──▶ 25 sequences / backbone
 [06,07] AlphaFold3, apo AND complex
 [10] filter:
        complex  pLDDT > 90, min PAE < 1.2, ipTM > 0.9, key-residue pLDDT > 85
        apo      pLDDT > 90, RMSD(apo, complex chain A) < 0.8 Å
```

### Scripts

| # | Script | Round | What it does |
|---|--------|-------|---------------|
| 00 | `00_build_catalytic_motifs.py` | setup | Enumerate His/Asp/Cys rotamer + substrate combinations into motif PDBs. |
| 01 | `01_rfdiffusion_taskgen.py` | 0 | All-atom RFdiffusion (`aa_ppi`) commands + SLURM array; **50,000 backbones**. |
| 02 | `02_filter_by_dssp.py` | 0 | Reject loop-dominated backbones via DSSP-style secondary-structure annotation. |
| 03 | `03_fix_motif_and_constraints.py` | 1, 2, 3 | Label the triad `FIXED`; write Rosetta constraints (triad + oxyanion hole + scissile peptide). *(PyRosetta)* |
| 04 | `04_pdbs_to_silent.py` | 1, 2, 3 | Convert FIXED PDBs to Rosetta silent files. |
| 05 | `05_fastrelax_mpnn_taskgen.py` | 1, 2, 3 | FastRelax + ProteinMPNN; `DESIGNS_PER_BACKBONE` = 15 / 40 / 25. |
| 06 | `06_af3_taskgen.py` | 1, 2, 3 | AlphaFold3 JSON + run commands. `MODE="apo"` every round; also `MODE="complex"` in round 3. |
| 07 | `07_collect_af3.py` | 1, 2, 3 | Collect AF3 confidences (mode-aware); computes `rmsd_to_mpnn` (apo) or `key_plddt` (complex). |
| 08 | `08_filter_apo_round.py` | 1, 2 | Apo confidence + fold-accuracy filter. |
| 09 | `09_partial_diffusion_taskgen.py` | 2, 3 | Partial-diffusion refinement; `PARTIAL_T`/`NUM_DESIGNS_PER_BACKBONE` = 40/15 then 10/30. |
| 10 | `10_filter_final_apo_complex.py` | 3 | Joint apo + complex confidence/agreement filter (final acceptance). |

Scripts 03–08 are reused across rounds by editing their `CONFIG` block
(`WORKING_DIR`, `INPUT_DIR`, `DESIGNS_PER_BACKBONE`, thresholds, ...) — they are
not duplicated per round.

### The catalytic-geometry constraint (stage 03)

FastRelax + ProteinMPNN is steered by a Rosetta constraint set
(`protease_pipeline/constraints.py`) that holds:

| Constraint | Atoms | Meaning |
|--------|-------|---------|
| `his_asp` | His ND1 ↔ Asp OD2/OD1 | Triad hydrogen bond |
| `his_cys` | His NE2 ↔ Cys SG | Base activates nucleophile |
| `his_N`   | His NE2 ↔ substrate P1′ N | Proton shuttle to leaving group |
| `oxy`     | Substrate P1 C=O ↔ oxyanion-hole amides | Transition-state stabilisation |
| `cys_C`   | Substrate P1 C ↔ Cys SG | Nucleophilic attack distance |
| triad angles/dihedral | Cys/His side-chain geometry | Triad orientation and planarity |

### The AF3 filters (stages 08, 10)

| Metric | Column | Round 1 | Round 2 | Round 3 (complex) | Round 3 (apo) |
|---|---|---|---|---|---|
| pLDDT | `plddt` / `complex_plddt` | > 80 | > 85 | > 90 | > 90 |
| pTM | `ptm` | > 0.8 | > 0.85 | – | – |
| ipTM | `iptm` | – | – | > 0.9 | – |
| min inter-chain PAE | `pae_interface` | – | – | < 1.2 | – |
| catalytic-residue pLDDT | `key_plddt` | – | – | > 85 | – |
| CA-RMSD to FastRelax/MPNN design | `rmsd_to_mpnn` | < 1.8 | < 1.2 | – | – |
| CA-RMSD, apo vs. complex chain A | `rmsd_apo_to_complex` | – | – | – | < 0.8 |

---

## Repository layout

```
protease-design-pipeline/
├── protease_pipeline/          # shared, importable helpers
│   ├── pdb_utils.py            #   PDB/mmCIF parsing, catalytic-residue lookup
│   ├── geometry.py             #   distances, angles, dihedrals, chain-aware CA-RMSD
│   ├── constraints.py          #   Rosetta constraint-file generation
│   └── slurm.py                #   SLURM array-script rendering
├── scripts/                    # numbered, per-stage pipeline scripts (00-10)
├── demo/                       # small simulated dataset + walkthrough (no external tools needed)
├── requirements.txt / environment.yml / pyproject.toml
├── CITATION.cff
└── LICENSE
```

---

## Installation

The Python glue in this repo is lightweight:

```bash
# conda
conda env create -f environment.yml
conda activate protease-design-pipeline

# or pip
pip install -r requirements.txt
# optional: install the helper package so `import protease_pipeline` works anywhere
pip install -e .
```

`protease_pipeline` requires numpy, pandas, biopython, matplotlib, seaborn.
Scripts also add the repo root to `sys.path` at runtime, so they import the
helper package without needing the editable install.

`03_fix_motif_and_constraints.py` additionally needs **PyRosetta**
(licence required):

```bash
conda install -c https://conda.rosettacommons.org pyrosetta
```

---

## Environment setup

**Nothing in this repo runs out of the box.** Every path to a container, model
checkpoint, or helper tool is a placeholder in angle brackets. Before running a
script, open its `CONFIG` block and fill these in for your site:

| Placeholder | Appears in | Fill in with |
|---|---|---|
| `RFDIFFUSION_SCRIPT`, `RFDIFFUSION_CHECKPOINT` | 01, 09 | Your all-atom RFdiffusion checkout + model weights |
| `UTIL_PATHS` | 02 | Directory containing RFdiffusion's `pdb_util`/`calc_dssp` modules |
| `SILENT_TOOL` | 04 | `silent_tools/silentfrompdbsparallel` from dl_binder_design |
| `MPNN_FR_CONTAINER`, `MPNN_FR_SCRIPT`, `MPNN_FR_CHECKPOINT` | 05 | Your ProteinMPNN + FastRelax (dl_binder_design `mpnn_fr`) setup |
| `AF3_CONTAINER`, `AF3_ENTRY` | 06 | Your AlphaFold3 installation |
| `PARTITION`, `GRES` | 01, 05, 06, 09 | Your Slurm partition / GPU resource strings |

The `#SBATCH` resource directives (memory, wall time, CPUs) reflect a Slurm
cluster and are otherwise reasonable starting points to tune for your queue.
`MAIL_USER` defaults to `None` (no `--mail-user` directive) so no identity is
embedded in generated submission scripts.

---

## Usage

Each stage script has a `CONFIG` block at the top. **Edit the paths and
parameters there, then run the script.**

```bash
python scripts/00_build_catalytic_motifs.py
python scripts/01_rfdiffusion_taskgen.py            # writes a SLURM array script
sbatch  diffusion/round0_scaffold/submit_rfdiffusion.sbatch
# ... once RFdiffusion has run:
python scripts/02_filter_by_dssp.py

# Round 1 (edit CONFIG in each script to point at round-1 directories first)
python scripts/03_fix_motif_and_constraints.py
python scripts/04_pdbs_to_silent.py
bash    design/round1_mpnn_fr/cmds_make_silent_file
python scripts/05_fastrelax_mpnn_taskgen.py          # DESIGNS_PER_BACKBONE = 15
sbatch  design/round1_mpnn_fr/submit_mpnn_fr.sbatch
python scripts/06_af3_taskgen.py                     # MODE = "apo"
sbatch  af3/round1_apo/submit_af3.sh
python scripts/07_collect_af3.py                     # MODE = "apo"
python scripts/08_filter_apo_round.py                # thresholds: 1.8 / 80 / 0.8

# Round 2: same steps with DESIGNS_PER_BACKBONE = 40 and thresholds 1.2 / 85 / 0.85,
# preceded by:
python scripts/09_partial_diffusion_taskgen.py       # PARTIAL_T=40, x15/backbone

# Round 3: same steps with DESIGNS_PER_BACKBONE = 25, preceded by:
python scripts/09_partial_diffusion_taskgen.py       # PARTIAL_T=10, x30/backbone
# then run 06/07 twice (MODE="apo" and MODE="complex") into separate working
# directories, and finish with:
python scripts/10_filter_final_apo_complex.py
```

Stages that drive a GPU/CPU model do not run it directly — they **generate a task
list** (one command per line) plus a **SLURM array submission script** that you
then `sbatch`. Collection stages parse the model outputs back into CSV tables.

---

## External tools

This pipeline orchestrates the following third-party tools. Install and cite them
per their own instructions; they are **not** bundled here.

| Tool | Used for | Source |
|------|----------|--------|
| RFdiffusion2-MI | Backbone scaffolding & partial diffusion | https://github.com/magnusbauer/RFDiffusion2_all_the_code |
| ProteinMPNN | Sequence design | https://github.com/dauparas/ProteinMPNN |
| dl_binder_design (FastRelax+MPNN, silent_tools) | Constrained sequence design + silent I/O | https://github.com/nrbennet/dl_binder_design |
| AlphaFold3 | Structure prediction (apo & complex) | https://github.com/google-deepmind/alphafold3 |
| PyRosetta | Constraint files & residue labelling | https://www.pyrosetta.org |

### Key references

- Bauer, M. S. et al. De novo design of phospho-tyrosine peptide binders. Preprint at https://doi.org/10.1101/2025.09.29.678898 (2025).
- Dauparas, J. et al. *Robust deep learning–based protein sequence design using
  ProteinMPNN.* **Science** (2022).
- Bennett, N. R. et al. *Improving de novo protein binder design with deep
  learning.* **Nature Communications** (2023). *(FastRelax + MPNN, silent_tools)*
- Abramson, J. et al. *Accurate structure prediction of biomolecular interactions
  with AlphaFold3.* **Nature** (2024).

*(Please confirm volumes/DOIs against the publisher of record before submission.)*

---

## Citing this work

See [`CITATION.cff`](CITATION.cff). The manuscript reference will be added on
publication.

## License

[MIT](LICENSE) © 2026 Hojae Choi
