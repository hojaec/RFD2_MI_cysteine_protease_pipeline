"""Rosetta constraint files encoding the catalytic geometry.

The constraints hold the Cys-His-Asp triad in a productive arrangement and
position the scissile peptide of the substrate against the nucleophile and the
oxyanion hole during ProteinMPNN + FastRelax.

Residue indices are Rosetta *pose* numbering (chains concatenated). The
substrate (chain B) therefore starts at ``chain_a_length + 1``; its scissile
P1 carbonyl carbon and the P1' leaving-group nitrogen sit at fixed offsets into
chain B (defaults 8 and 9, matching the substrate register used in this project).
"""

from __future__ import annotations

CST_TEMPLATE = """\
AtomPair ND1 {his} OD2 {asp} HARMONIC 3.0 0.1
AtomPair ND1 {his} OD1 {asp} HARMONIC 3.2 0.1
AtomPair NE2 {his} SG {cys} HARMONIC 3.5 0.1
AtomPair NE2 {his} N {lg_n} HARMONIC 4.0 0.1
AtomPair O {p1_c} N {cys_minus_2} HARMONIC 2.8 0.1
AtomPair O {p1_c} N {cys} HARMONIC 3.0 0.1
AtomPair C {p1_c} SG {cys} HARMONIC 3.0 0.1
Angle CG {his} SG {cys} CB {cys} HARMONIC 1.55 0.08
Angle NE2 {his} SG {cys} CB {cys} HARMONIC 1.54 0.08
Angle CE1 {his} NE2 {his} SG {cys} HARMONIC 1.42 0.08
Angle SG {cys} C {p1_c} O {p1_c} HARMONIC 1.55 0.03
Dihedral CE1 {his} NE2 {his} SG {cys} CB {cys} HARMONIC 1.65 0.1
"""


def catalytic_triad_constraints(
    his,
    asp,
    cys,
    chain_a_length,
    p1_carbonyl_offset=8,
    leaving_group_offset=9,
):
    """Return the constraint-file text for one design (Rosetta pose numbering)."""
    return CST_TEMPLATE.format(
        his=his,
        asp=asp,
        cys=cys,
        cys_minus_2=cys - 2,
        p1_c=chain_a_length + p1_carbonyl_offset,
        lg_n=chain_a_length + leaving_group_offset,
    )
