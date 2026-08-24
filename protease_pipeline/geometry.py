"""Geometric measurements used by the catalytic-geometry filters.

All angle/dihedral routines take numpy coordinate vectors (shape ``(3,)``) and
return degrees. They mirror the definitions used throughout the design
notebook so that filter thresholds stay comparable across runs.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB import PDBParser, Superimposer


def distance(coord1, coord2):
    """Euclidean distance between two points, or ``None`` if either is missing."""
    if coord1 is None or coord2 is None:
        return None
    return float(np.linalg.norm(np.asarray(coord1) - np.asarray(coord2)))


def calculate_angle(coord1, coord2, coord3):
    """Angle (degrees) at ``coord2`` formed by coord1-coord2-coord3."""
    if any(c is None for c in (coord1, coord2, coord3)):
        return None
    vec1 = np.asarray(coord1) - np.asarray(coord2)
    vec2 = np.asarray(coord3) - np.asarray(coord2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return None
    cosine_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_dihedral(coord1, coord2, coord3, coord4):
    """Signed dihedral (degrees) about the coord2-coord3 axis."""
    if any(c is None for c in (coord1, coord2, coord3, coord4)):
        return None
    b1 = np.asarray(coord2) - np.asarray(coord1)
    b2 = np.asarray(coord3) - np.asarray(coord2)
    b3 = np.asarray(coord4) - np.asarray(coord3)

    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    b3 = b3 / np.linalg.norm(b3)

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    m1 = np.cross(n1, b2)
    x = np.clip(np.dot(n1, n2), -1.0, 1.0)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))


def get_ca_atoms(structure, chain=None):
    """Return CA atoms in a Bio.PDB structure, in order.

    If ``chain`` is given, only that chain ID is included — needed to compare
    a single-chain (apo) prediction against chain A of a multi-chain model.
    """
    return [
        residue["CA"]
        for model in structure
        for ch in model
        for residue in ch
        if residue.has_id("CA") and (chain is None or ch.id == chain)
    ]


def calculate_ca_rmsd(pdb_path1, pdb_path2, chain1=None, chain2=None):
    """Superposition CA-RMSD between two PDB files.

    ``chain1``/``chain2`` restrict each structure to one chain ID before
    comparing (e.g. to compare an apo chain-A prediction against chain A of a
    two-chain complex model). Returns ``None`` if either file fails to parse
    or the selected CA counts differ.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure1 = parser.get_structure("s1", pdb_path1)
        structure2 = parser.get_structure("s2", pdb_path2)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        print(f"Error parsing PDB files: {pdb_path1}, {pdb_path2}: {exc}")
        return None

    ca1 = get_ca_atoms(structure1, chain=chain1)
    ca2 = get_ca_atoms(structure2, chain=chain2)
    if len(ca1) != len(ca2):
        print(
            f"Warning: CA count differs\n  {pdb_path1}: {len(ca1)}\n  {pdb_path2}: {len(ca2)}"
        )
        return None

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    return float(sup.rms)
