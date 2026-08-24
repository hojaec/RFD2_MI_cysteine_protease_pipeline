"""Structure-parsing helpers shared across pipeline stages.

Covers the two structure conventions used in this project:

* Design/prediction models are two-chain complexes: chain ``A`` is the designed
  protease, chain ``B`` is the bound peptide substrate.
* The catalytic machinery is a Cys-His-Asp triad located in chain ``A``.
"""

from __future__ import annotations

import os
import re

from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CATALYTIC_RESNAMES = frozenset({"CYS", "HIS", "ASP"})


# --------------------------------------------------------------------------- #
# Catalytic-residue lookup
# --------------------------------------------------------------------------- #
def key_residues_from_chainA(pdb_file, resnames=CATALYTIC_RESNAMES):
    """Residue numbers in chain A whose residue name is in ``resnames``.

    Reads ATOM records directly (robust to non-standard files). Returns a set of
    integer residue numbers.
    """
    key_residues = set()
    try:
        with open(pdb_file) as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                if line[21].strip() != "A":
                    continue
                if line[17:20].strip() in resnames:
                    try:
                        key_residues.add(int(line[22:26].strip()))
                    except ValueError:
                        continue
    except OSError as exc:
        print(f"Error reading template pdb file {pdb_file}: {exc}")
    return key_residues


def extract_first_catalytic_residues(pdb_file):
    """Return ``(cys_resnum, his_resnum, asp_resnum)`` for the first of each in chain A.

    Uses Biopython and skips hetero residues. Any triad member not found is
    returned as ``None``.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", pdb_file)
    cys_residue = his_residue = asp_residue = None
    for model in structure:
        if "A" not in model:
            continue
        for residue in model["A"]:
            if residue.id[0] != " ":  # skip hetero/water
                continue
            name = residue.get_resname()
            resid = residue.get_id()[1]
            if name == "CYS" and cys_residue is None:
                cys_residue = resid
            elif name == "HIS" and his_residue is None:
                his_residue = resid
            elif name == "ASP" and asp_residue is None:
                asp_residue = resid
            if cys_residue and his_residue and asp_residue:
                return cys_residue, his_residue, asp_residue
        break
    return cys_residue, his_residue, asp_residue


def get_non_alanine_residues(pdb_file):
    """Sorted ``chain+resnum`` keys (e.g. ``"A145"``) of all non-alanine residues.

    Used to build the ``--fixed_residues`` set for MPNN so only alanine
    positions (the diffusion placeholder) are redesigned.
    """
    parser = PDBParser(QUIET=True)
    fixed = set()
    structure = parser.get_structure("s", pdb_file)
    for chain in structure[0]:
        for res in chain:
            if res.get_id()[0] != " ":  # standard residues only
                continue
            if res.get_resname().strip() != "ALA":
                fixed.add(f"{chain.get_id()}{res.get_id()[1]}")
    return sorted(fixed)


# --------------------------------------------------------------------------- #
# Sequence extraction
# --------------------------------------------------------------------------- #
def extract_chain_sequences(pdb_file, chains=("A", "B")):
    """Map ``chain_id -> one-letter sequence`` using Biopython's PPBuilder."""
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    sequences = {}
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    model = structure[0]
    for chain_id in chains:
        if chain_id not in model:
            print(f"Warning: chain {chain_id} not found in {pdb_file}")
            continue
        polys = ppb.build_peptides(model[chain_id])
        if polys:
            sequences[chain_id] = "".join(str(pp.get_sequence()) for pp in polys)
        else:
            print(f"Warning: no polypeptide in chain {chain_id} of {pdb_file}")
    return sequences


def extract_sequence_from_atom_records(pdb_file, chain="A"):
    """One-letter sequence for a chain, read straight from ATOM records.

    Preserves residue order and de-duplicates by residue number. Used where a
    dependency-light parser is preferred (e.g. Chai-1 input generation).
    """
    sequence = []
    seen = set()
    with open(pdb_file) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain:
                continue
            resnum = line[22:26].strip()
            if resnum in seen:
                continue
            seen.add(resnum)
            resname = line[17:20].strip()
            if resname in THREE_TO_ONE:
                sequence.append(THREE_TO_ONE[resname])
    return "".join(sequence)


def get_chain_start_residue(pdb_file, chain="B"):
    """First residue number of ``chain`` in a PDB file, or ``None``."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_file)
    for ch in structure.get_chains():
        if ch.id == chain:
            for residue in ch.get_residues():
                return residue.id[1]
    return None


def get_atom_coordinates(structure, chain_id, residue_number, atom_name):
    """Coordinate of a named atom, or ``None`` if the atom/residue is absent."""
    try:
        return structure[0][chain_id][residue_number][atom_name].coord
    except KeyError:
        print(
            f"Atom {atom_name} / residue {residue_number} in chain {chain_id} not found."
        )
        return None


# --------------------------------------------------------------------------- #
# Format conversion
# --------------------------------------------------------------------------- #
def cif_to_pdb(cif_path, pdb_path):
    """Convert an mmCIF file to PDB. Returns True on success."""
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()
    try:
        structure = parser.get_structure("structure", cif_path)
        io.set_structure(structure)
        io.save(pdb_path)
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        print(f"Error converting {cif_path}: {exc}")
        return False


def parse_mmcif_atoms(file_path):
    """Ordered list of ``(chain, resseq, resname, atom_name)`` from an mmCIF file.

    A minimal ``_atom_site`` loop reader that preserves the file's atom order so
    it can be indexed against AlphaFold3 ``atom_plddts`` arrays. Prefers
    ``auth_*`` columns and falls back to ``label_*``.
    """
    atoms = []
    try:
        with open(file_path) as handle:
            lines = handle.readlines()
    except OSError as exc:
        print(f"Error parsing mmCIF file {file_path}: {exc}")
        return atoms

    def _find(headers, *names):
        for name in names:
            if name in headers:
                return headers.index(name)
        return None

    for i, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        j = i + 1
        headers = []
        while j < len(lines) and lines[j].startswith("_atom_site."):
            headers.append(lines[j].strip())
            j += 1
        if not headers:
            continue

        chain_index = _find(headers, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        resseq_index = _find(headers, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        resname_index = _find(headers, "_atom_site.auth_comp_id", "_atom_site.label_comp_id")
        atom_name_index = _find(headers, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        if None in (chain_index, resseq_index, resname_index, atom_name_index):
            print(f"Required _atom_site columns not found in {file_path}")
            return atoms

        needed = max(chain_index, resseq_index, resname_index, atom_name_index) + 1
        for row in lines[j:]:
            if row.startswith("loop_") or row.strip() == "":
                break
            parts = row.split()
            if len(parts) < needed:
                continue
            resseq_str = parts[resseq_index]
            try:
                resseq = int(resseq_str)
            except ValueError:
                digits = re.findall(r"\d+", resseq_str)
                if not digits:
                    continue
                resseq = int(digits[0])
            atoms.append(
                (parts[chain_index], resseq, parts[resname_index], parts[atom_name_index])
            )
        break  # only the first _atom_site loop
    return atoms
