#!/usr/bin/env python3
"""Generate the small simulated dataset used by the demo/ walkthrough.

Everything this script writes is **synthetic**: idealized poly-alanine-like
backbones placed with the NeRF algorithm (correct bond lengths/angles, so
Biopython's peptide builder and CA superposition behave sensibly) with a
His/Asp/Cys triad substituted in at fixed positions. These are stand-ins for
real RFdiffusion/FastRelax+MPNN/AlphaFold3 output, built only so the *code*
(motif assembly, AF3-confidence collection, and the round filters) can be
exercised end-to-end without any of those external tools installed. Do not
read anything structural or biological into the coordinates.

Running this script regenerates every file under demo/ from scratch:

    python demo/generate_demo_data.py

See demo/README.md for what each generated case is meant to demonstrate.
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Idealized backbone geometry (Engh & Huber bond lengths/angles) placed with
# the NeRF (natural extension reference frame) algorithm.
# --------------------------------------------------------------------------- #
N_CA, CA_C, C_N, C_O, CA_CB = 1.458, 1.525, 1.329, 1.231, 1.530
ANG_N_CA_C = math.radians(111.2)
ANG_CA_C_N = math.radians(117.2)
ANG_C_N_CA = math.radians(121.7)
ANG_CA_C_O = math.radians(120.8)
OMEGA = math.pi  # trans peptide bond

PHI_HELIX, PSI_HELIX = math.radians(-57.0), math.radians(-47.0)
PHI_STRAND, PSI_STRAND = math.radians(-120.0), math.radians(140.0)


def _nerf(a, b, c, bond_length, bond_angle, dihedral):
    """Place a 4th atom given three previous atoms and internal coordinates."""
    bc_hat = (c - b) / np.linalg.norm(c - b)
    n_hat = np.cross(b - a, bc_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)
    m_hat = np.cross(n_hat, bc_hat)
    local = np.array([
        -bond_length * math.cos(bond_angle),
        bond_length * math.sin(bond_angle) * math.cos(dihedral),
        bond_length * math.sin(bond_angle) * math.sin(dihedral),
    ])
    basis = np.array([bc_hat, m_hat, n_hat]).T
    return c + basis.dot(local)


def build_backbone(n_res, phi, psi):
    """N/CA/C coordinates for ``n_res`` residues of constant (phi, psi)."""
    n0 = np.array([0.0, 0.0, 0.0])
    ca0 = n0 + np.array([N_CA, 0.0, 0.0])
    theta = math.pi - ANG_N_CA_C
    c0 = ca0 + CA_C * np.array([math.cos(theta), math.sin(theta), 0.0])
    coords = [n0, ca0, c0]
    for _ in range(1, n_res):
        prev_n, prev_ca, prev_c = coords[-3], coords[-2], coords[-1]
        new_n = _nerf(prev_n, prev_ca, prev_c, C_N, ANG_CA_C_N, psi)
        new_ca = _nerf(prev_ca, prev_c, new_n, N_CA, ANG_C_N_CA, OMEGA)
        new_c = _nerf(prev_c, new_n, new_ca, CA_C, ANG_N_CA_C, phi)
        coords += [new_n, new_ca, new_c]
    return [coords[i:i + 3] for i in range(0, len(coords), 3)]  # per-residue [N, CA, C]


def carbonyl_oxygen(n, ca, c, psi):
    return _nerf(n, ca, c, C_O, ANG_CA_C_O, psi + math.pi)


def beta_carbon(n, ca, c):
    """Tetrahedral-ish CB off the N-CA-C frame (chirality not enforced)."""
    n1 = (n - ca) / np.linalg.norm(n - ca)
    c1 = (c - ca) / np.linalg.norm(c - ca)
    bisector = -(n1 + c1)
    bisector /= np.linalg.norm(bisector)
    perp = np.cross(n1, c1)
    perp /= np.linalg.norm(perp)
    ang = math.radians(54.75)
    direction = bisector * math.cos(ang) + perp * math.sin(ang)
    direction /= np.linalg.norm(direction)
    return ca + CA_CB * direction


def extend(base, direction_from, length, twist=0.0, ref=None):
    """Walk one more bond of ``length`` away from ``base`` along direction_from->base,
    with an optional twist around that axis using ``ref`` to define the swing plane."""
    axis = (base - direction_from) / np.linalg.norm(base - direction_from)
    if ref is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(axis, ref)
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)
    direction = axis * math.cos(math.radians(70)) + (
        perp * math.cos(twist) + perp2 * math.sin(twist)
    ) * math.sin(math.radians(70))
    direction /= np.linalg.norm(direction)
    return base + length * direction


ELEMENT = {
    "N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C",
    "CG": "C", "SG": "S", "OD1": "O", "OD2": "O",
    "ND1": "N", "CE1": "C", "NE2": "N", "CD2": "C",
}


def pdb_line(serial, name, resname, chain, resseq, xyz):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:<4}{'':1}{resname:>3} {chain:1}{resseq:4d}{'':1}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}"
        f"{'':10}{ELEMENT.get(name, 'C'):>2}"
    )


class ChainBuilder:
    """Builds one idealized chain, substituting a His/Asp/Cys triad at given positions."""

    def __init__(self, chain_id, n_res, phi, psi, start_resseq=1, triad=None):
        self.chain_id = chain_id
        self.n_res = n_res
        self.phi, self.psi = phi, psi
        self.start_resseq = start_resseq
        self.triad = triad or {}  # {resseq: "HIS"|"ASP"|"CYS"}
        self.residues = build_backbone(n_res, phi, psi)
        self.lines = []
        self.serial = 0
        self._build()

    def _add(self, name, resname, resseq, xyz):
        self.serial += 1
        self.lines.append(pdb_line(self.serial, name, resname, self.chain_id, resseq, xyz))

    def _build(self):
        for i, (n, ca, c) in enumerate(self.residues):
            resseq = self.start_resseq + i
            resname = self.triad.get(resseq, "ALA")
            o = carbonyl_oxygen(n, ca, c, self.psi)
            self._add("N", resname, resseq, n)
            self._add("CA", resname, resseq, ca)
            self._add("C", resname, resseq, c)
            self._add("O", resname, resseq, o)
            if resname == "ALA":
                cb = beta_carbon(n, ca, c)
                self._add("CB", resname, resseq, cb)
                continue

            cb = beta_carbon(n, ca, c)
            self._add("CB", resname, resseq, cb)
            if resname == "CYS":
                sg = extend(cb, ca, 1.81, twist=0.3)
                self._add("SG", resname, resseq, sg)
            elif resname == "ASP":
                cg = extend(cb, ca, 1.52, twist=0.1)
                od1 = extend(cg, cb, 1.25, twist=0.4)
                od2 = extend(cg, cb, 1.25, twist=-2.6)
                self._add("CG", resname, resseq, cg)
                self._add("OD1", resname, resseq, od1)
                self._add("OD2", resname, resseq, od2)
            elif resname == "HIS":
                cg = extend(cb, ca, 1.50, twist=0.1)
                nd1 = extend(cg, cb, 1.38, twist=0.4)
                cd2 = extend(cg, cb, 1.36, twist=-2.4)
                ce1 = extend(nd1, cg, 1.32, twist=1.6)
                ne2 = extend(cd2, cg, 1.35, twist=-1.6)
                self._add("CG", resname, resseq, cg)
                self._add("ND1", resname, resseq, nd1)
                self._add("CD2", resname, resseq, cd2)
                self._add("CE1", resname, resseq, ce1)
                self._add("NE2", resname, resseq, ne2)

    def text(self):
        return "\n".join(self.lines) + "\n"

    def atoms(self):
        """Ordered (chain, resseq, resname, atom_name, xyz) tuples, for CIF export."""
        out = []
        for line in self.lines:
            name = line[12:16].strip()
            resname = line[17:20].strip()
            resseq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            out.append((self.chain_id, resseq, resname, name, (x, y, z)))
        return out


def jitter(builder, scale=0.05, seed=0):
    """Small per-atom coordinate noise (simulates prediction-to-prediction wobble)."""
    rng = np.random.RandomState(seed)
    new = ChainBuilder.__new__(ChainBuilder)
    new.__dict__.update(builder.__dict__)
    new_lines = []
    for line in builder.lines:
        x = float(line[30:38]) + rng.normal(0, scale)
        y = float(line[38:46]) + rng.normal(0, scale)
        z = float(line[46:54]) + rng.normal(0, scale)
        new_lines.append(line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
    new.lines = new_lines
    return new


def bend_chain_a(chain_a):
    """Rebuild chain A's backbone as an extended strand instead of a helix.

    Same residues/atom count as ``chain_a``, but a very different fold - used
    to simulate an apo prediction whose CA-RMSD to a reference is large.
    """
    bent = ChainBuilder.__new__(ChainBuilder)
    bent.__dict__.update(chain_a.__dict__)
    bent.phi, bent.psi = PHI_STRAND, PSI_STRAND
    bent.residues = build_backbone(chain_a.n_res, PHI_STRAND, PSI_STRAND)
    bent.lines, bent.serial = [], 0
    ChainBuilder._build(bent)
    return bent


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def two_chain_pdb(chain_a, chain_b):
    return chain_a.text() + "TER\n" + chain_b.text() + "TER\nEND\n"


def confidences_json(path, n_atoms, plddt_value, rng):
    atom_plddts = list(np.clip(rng.normal(plddt_value, 1.5, n_atoms), 0, 100))
    write(path, json.dumps({"atom_plddts": atom_plddts}))
    return atom_plddts


def summary_apo_json(path, ptm):
    write(path, json.dumps({"ptm": ptm}))


def minimal_cif(path, atoms):
    header = (
        "data_demo\n#\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
    )
    rows = []
    for i, (chain, resseq, resname, name, (x, y, z)) in enumerate(atoms, start=1):
        elem = ELEMENT.get(name, "C")
        rows.append(f"ATOM {i} {elem} {name} {resname} {chain} {resseq} {x:.3f} {y:.3f} {z:.3f}")
    write(path, header + "\n".join(rows) + "\n")


# --------------------------------------------------------------------------- #
# Stage 00 demo: standalone rotamer + target fragments (assembled by
# scripts/00_build_catalytic_motifs.py via plain text concatenation).
# --------------------------------------------------------------------------- #
def build_stage00():
    his = ChainBuilder("A", 1, PHI_HELIX, PSI_HELIX, start_resseq=50, triad={50: "HIS"})
    asp = ChainBuilder("A", 1, PHI_HELIX, PSI_HELIX, start_resseq=80, triad={80: "ASP"})
    cys = ChainBuilder("A", 1, PHI_HELIX, PSI_HELIX, start_resseq=120, triad={120: "CYS"})
    target = ChainBuilder("B", 12, PHI_STRAND, PSI_STRAND, start_resseq=1)

    write(os.path.join(DEMO_DIR, "motifs/rotamers/his_1.pdb"), his.text())
    write(os.path.join(DEMO_DIR, "motifs/rotamers/asp_1.pdb"), asp.text())
    write(os.path.join(DEMO_DIR, "motifs/rotamers/cys_1.pdb"), cys.text())
    write(os.path.join(DEMO_DIR, "motifs/rotamers/target_1.pdb"), target.text())


# --------------------------------------------------------------------------- #
# Stage 07/08 demo (round 1): FastRelax+MPNN "design models" plus a simulated
# AlphaFold3 apo output tree, with one design that passes the round-1 filter,
# one rejected for high RMSD-to-design, and one rejected for low confidence.
# --------------------------------------------------------------------------- #
TRIAD_A = {6: "HIS", 10: "ASP", 15: "CYS"}
CHAIN_A_LEN = 18
CHAIN_B_LEN = 12


def _design_model(name_seed):
    chain_a = ChainBuilder("A", CHAIN_A_LEN, PHI_HELIX, PSI_HELIX, start_resseq=1, triad=TRIAD_A)
    chain_b = ChainBuilder("B", CHAIN_B_LEN, PHI_STRAND, PSI_STRAND, start_resseq=1)
    return chain_a, chain_b


def build_round1_demo():
    rng = np.random.RandomState(2)
    design_dir = os.path.join(DEMO_DIR, "design/round1_mpnn_fr/pdbs")
    af3_root = os.path.join(DEMO_DIR, "af3/round1_apo")

    cases = [
        # (stem, apo prediction builder, plddt, ptm)
        ("helixA", "near_identical", 88.0, 0.86),
        ("helixB", "shifted", 84.0, 0.84),
        ("helixC", "near_identical", 58.0, 0.52),
    ]

    for stem, kind, plddt_val, ptm in cases:
        chain_a, chain_b = _design_model(stem)
        write(os.path.join(design_dir, f"{stem}.pdb"), two_chain_pdb(chain_a, chain_b))

        if kind == "near_identical":
            apo_pred = jitter(chain_a, scale=0.08, seed=hash(stem) % (2**31))
        else:  # "shifted" - refolded backbone -> high CA-RMSD after superposition
            apo_pred = bend_chain_a(chain_a)

        description = f"{stem}_dldesign_0_cycle1"
        sample_dir = os.path.join(af3_root, "output", description, "seed-1_sample-1")
        os.makedirs(sample_dir, exist_ok=True)

        atoms = apo_pred.atoms()
        confidences_json(
            os.path.join(sample_dir, "confidences.json"), len(atoms), plddt_val, rng
        )
        summary_apo_json(os.path.join(sample_dir, "summary_confidences.json"), ptm)
        minimal_cif(os.path.join(sample_dir, "model.cif"), atoms)
        # 07_collect_af3.py expects the descriptively-named PDB already in place
        # (AlphaFold3 + this lab's post-processing emit it alongside model.cif).
        named_pdb = os.path.join(
            sample_dir, f"{description}_seed-1_sample-1_model.pdb"
        )
        write(named_pdb, apo_pred.text())


# --------------------------------------------------------------------------- #
# Stage 10 demo (round 3): pre-collected apo + complex AF3 tables (as produced
# by two separate runs of scripts/07_collect_af3.py) plus the PDB files their
# pdb_path columns point to, covering: accepted, rejected for apo/complex
# disagreement, and rejected for low confidence.
# --------------------------------------------------------------------------- #
def build_round3_demo():
    apo_dir = os.path.join(DEMO_DIR, "af3/round3_apo/pdbs")
    complex_dir = os.path.join(DEMO_DIR, "af3/round3_complex/pdbs")

    apo_rows, complex_rows = [], []

    # 1) candidate_01: confident and self-consistent -> ACCEPTED
    chain_a, chain_b = _design_model("candidate_01")
    complex_pdb = os.path.join(complex_dir, "candidate_01_complex.pdb")
    apo_pdb = os.path.join(apo_dir, "candidate_01_apo.pdb")
    write(complex_pdb, two_chain_pdb(chain_a, chain_b))
    write(apo_pdb, jitter(chain_a, scale=0.05, seed=10).text())
    apo_rows.append({"description": "candidate_01", "pdb_path": apo_pdb, "plddt": 92.5})
    complex_rows.append({
        "description": "candidate_01", "pdb_path": complex_pdb,
        "complex_plddt": 93.1, "pae_interface": 0.85, "iptm": 0.93, "key_plddt": 88.4,
    })

    # 2) candidate_02_induced_fit: confident and passes both thresholds individually,
    #    but the apo (unbound) fold disagrees with the complex's chain A -> REJECTED
    #    by the apo/complex agreement check.
    chain_a2, chain_b2 = _design_model("candidate_02")
    complex_pdb2 = os.path.join(complex_dir, "candidate_02_complex.pdb")
    apo_pdb2 = os.path.join(apo_dir, "candidate_02_apo.pdb")
    write(complex_pdb2, two_chain_pdb(chain_a2, chain_b2))
    write(apo_pdb2, bend_chain_a(chain_a2).text())
    apo_rows.append({"description": "candidate_02_induced_fit", "pdb_path": apo_pdb2, "plddt": 91.0})
    complex_rows.append({
        "description": "candidate_02_induced_fit", "pdb_path": complex_pdb2,
        "complex_plddt": 92.0, "pae_interface": 1.00, "iptm": 0.91, "key_plddt": 86.0,
    })

    # 3) candidate_03_low_confidence: apo pLDDT and complex ipTM both miss threshold
    #    -> REJECTED before the RMSD/agreement check is even relevant.
    chain_a3, chain_b3 = _design_model("candidate_03")
    complex_pdb3 = os.path.join(complex_dir, "candidate_03_complex.pdb")
    apo_pdb3 = os.path.join(apo_dir, "candidate_03_apo.pdb")
    write(complex_pdb3, two_chain_pdb(chain_a3, chain_b3))
    write(apo_pdb3, jitter(chain_a3, scale=0.05, seed=30).text())
    apo_rows.append({"description": "candidate_03_low_confidence", "pdb_path": apo_pdb3, "plddt": 85.0})
    complex_rows.append({
        "description": "candidate_03_low_confidence", "pdb_path": complex_pdb3,
        "complex_plddt": 91.0, "pae_interface": 1.10, "iptm": 0.85, "key_plddt": 87.0,
    })

    write_csv(
        os.path.join(DEMO_DIR, "af3/round3_apo/af3_data.csv"),
        apo_rows, ["description", "pdb_path", "plddt"],
    )
    write_csv(
        os.path.join(DEMO_DIR, "af3/round3_complex/af3_data.csv"),
        complex_rows,
        ["description", "pdb_path", "complex_plddt", "pae_interface", "iptm", "key_plddt"],
    )


def main():
    build_stage00()
    build_round1_demo()
    build_round3_demo()
    print(f"Demo dataset written under {DEMO_DIR}")


if __name__ == "__main__":
    main()
