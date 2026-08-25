# Demo: running the pipeline's code on a small simulated dataset

This directory lets you run the parts of the pipeline that are **pure
Python/pandas** — no RFdiffusion, PyRosetta, ProteinMPNN/FastRelax, or
AlphaFold3 required — against a small, clearly synthetic dataset, so you can
confirm the orchestration/filtering code itself works before setting up any
of the external tools.

**Everything under `demo/` is simulated.** `generate_demo_data.py` builds
idealized poly-alanine-like backbones (placed with the NeRF algorithm, so
bond lengths/angles are sane and Biopython's peptide builder and CA
superposition behave correctly) with a His/Asp/Cys triad substituted in at
fixed positions, plus hand-written AlphaFold3-style `confidences.json` /
`summary_confidences.json` files. None of it comes from a real diffusion,
design, or structure-prediction run. Regenerate it at any time with:

```bash
python demo/generate_demo_data.py
```

## What's covered — and what isn't

| Script | Runs in this demo? | Why |
|---|---|---|
| `00_build_catalytic_motifs.py` | ✅ | Pure file concatenation |
| `01_rfdiffusion_taskgen.py` | ❌ | Needs an RFdiffusion checkout + weights |
| `02_filter_by_dssp.py` | ❌ | Needs RFdiffusion's `pdb_util`/`calc_dssp` |
| `03_fix_motif_and_constraints.py` | ❌ | Needs PyRosetta |
| `04_pdbs_to_silent.py` | ❌ | Needs dl_binder_design's `silent_tools` |
| `05_fastrelax_mpnn_taskgen.py` | ❌ | Needs the MPNN+FastRelax driver |
| `06_af3_taskgen.py` | ❌ | Needs an AlphaFold3 installation |
| `07_collect_af3.py` (apo) | ✅ | Pure pandas/Biopython parsing |
| `08_filter_apo_round.py` | ✅ | Pure pandas |
| `09_partial_diffusion_taskgen.py` | ❌ | Needs RFdiffusion |
| `10_filter_final_apo_complex.py` | ✅ | Pure pandas + CA-RMSD |

`demo_library_functions.py` additionally calls a few `protease_pipeline`
helpers directly (triad lookup, fixed-residue set, sequence extraction,
Rosetta constraint-file generation, CA-RMSD) that are used by the
PyRosetta/MPNN stages above but aren't otherwise reachable without those
tools installed.

See the main [README](../README.md#environment-setup) for what each
placeholder needs once you do have those tools available.

## Layout

```
demo/
├── generate_demo_data.py          # (re)builds everything below
├── demo_library_functions.py      # calls protease_pipeline helpers directly
├── motifs/rotamers/                       # stage-00 input
├── design/round1_mpnn_fr/pdbs/            # "FastRelax+MPNN" design models (round 1)
├── af3/round1_apo/output/                 # simulated AF3 apo output tree (round 1)
├── af3/round3_apo/, af3/round3_complex/   # pre-collected round-3 apo/complex tables
```

The paths above are exactly the default `CONFIG` values in scripts 00, 07, 08,
and 10 — run the commands below from *inside* `demo/` and nothing needs
editing.

## Run it

```bash
cd demo

# Stage 00: assemble the one motif PDB from its rotamer + target fragments.
python ../scripts/00_build_catalytic_motifs.py
#   -> Wrote 1 motif PDBs to motifs/combined

# Stage 07 (apo) + 08: collect simulated AF3 confidences, apply round-1 filter.
# Three designs, three outcomes:
#   helixA -> passes (confident, low RMSD to its design model)
#   helixB -> rejected: RMSD to design model too high (~9.9 A)
#   helixC -> rejected: low pLDDT/pTM
python ../scripts/07_collect_af3.py
python ../scripts/08_filter_apo_round.py
#   -> Passing samples: 1 / 3

# Stage 10: joint apo + complex round-3 filter, using pre-collected tables
# (as if 07_collect_af3.py had already been run once in each MODE).
# Three candidates, three outcomes:
#   candidate_01               -> accepted
#   candidate_02_induced_fit    -> rejected: apo fold disagrees with the bound
#                                  complex's chain A (RMSD > 0.8 A) even though
#                                  both individually pass their confidence bars
#   candidate_03_low_confidence -> rejected: apo pLDDT and complex ipTM both
#                                  miss threshold
python ../scripts/10_filter_final_apo_complex.py
#   -> Designs passing the joint apo/complex filter: 1

# Bonus: exercise a few library helpers directly (no external tools needed).
python demo_library_functions.py
```

Each command prints how many rows/designs passed, and writes its output CSV
next to the input (`af3/round1_apo/af3_data.csv`,
`af3/round1_apo/af3_data_filtered.csv`, `motifs/combined/`,
`af3/round3_final/af3_data_filtered.csv`). Those are regenerated run output,
not part of the tracked demo dataset — delete them (or just re-run
`generate_demo_data.py`) to get back to a clean starting point.
