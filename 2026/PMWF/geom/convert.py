#!/usr/bin/env python3
"""
Parse a simple crystal-structure text file of the form:

$cell
  a b c alpha beta gamma
$coord
  x y z element
  ...
$end

Outputs:
  (i) 3x3 lattice-vector matrix (rows are a,b,c vectors)
  (ii) an XYZ file for the listed coordinates

Usage:
  python parse_cell_coord.py input.txt --xyz out.xyz --lat out_lattice.txt

Notes:
- Coordinates are written to XYZ exactly as given (assumed Cartesian, in the same units as the input).
- Lattice vectors are built from (a,b,c,alpha,beta,gamma) in degrees using a standard convention.
"""

import argparse
import math
import numpy as np
from typing import List, Tuple

from pyscf.data.nist import BOHR

from zflow.pyscf_helper import LAT


def cell_params_to_lattice(a: float, b: float, c: float,
                           alpha_deg: float, beta_deg: float, gamma_deg: float) -> List[List[float]]:
    """Return 3x3 lattice vectors (rows: a_vec, b_vec, c_vec) from cell parameters."""
    alpha = math.radians(alpha_deg)
    beta  = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_g = math.sin(gamma)

    if abs(sin_g) < 1e-14:
        raise ValueError("Invalid cell: sin(gamma) is ~0, cannot construct lattice vectors.")

    # Convention:
    # a = (a, 0, 0)
    # b = (b*cos(gamma), b*sin(gamma), 0)
    # c = (c*cos(beta),
    #      c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma),
    #      c*sqrt(1 - cos(beta)^2 - (...)^2 ) )
    a_vec = [a, 0.0, 0.0]
    b_vec = [b * cos_g, b * sin_g, 0.0]

    c_x = c * cos_b
    c_y = c * (cos_a - cos_b * cos_g) / sin_g

    # Guard against small negative due to floating error
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    if c_z_sq < -1e-10:
        raise ValueError(f"Invalid cell: computed c_z^2 is negative ({c_z_sq}). Check angles/lengths.")
    c_z = math.sqrt(max(0.0, c_z_sq))

    c_vec = [c_x, c_y, c_z]
    return np.asarray([a_vec, b_vec, c_vec])


def parse_structure(fcoord: str) -> Tuple[Tuple[float, float, float, float, float, float],
                                        List[Tuple[str, float, float, float]]]:
    """Parse $cell and $coord blocks. Returns (cell_params, atoms)."""
    with open(fcoord, 'r') as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    cell_params = None
    atoms: List[Tuple[str, float, float, float]] = []

    i = 0
    while i < len(lines):
        ln = lines[i].lower()
        if ln.startswith("$cell"):
            i += 1
            if i >= len(lines):
                raise ValueError("Found $cell but no cell-parameter line.")
            parts = lines[i].split()
            if len(parts) < 6:
                raise ValueError(f"Cell line must have 6 numbers: a b c alpha beta gamma. Got: {lines[i]}")
            a, b, c, alpha, beta, gamma = map(float, parts[:6])
            cell_params = (a, b, c, alpha, beta, gamma)
            alat = cell_params_to_lattice(*cell_params) * BOHR
            i += 1
            continue

        if ln.startswith("$coord"):
            i += 1
            while i < len(lines) and not lines[i].lower().startswith("$end"):
                parts = lines[i].split()
                if len(parts) < 4:
                    raise ValueError(f"Bad coord line (need x y z element): {lines[i]}")
                x, y, z = map(float, parts[:3])
                x *= BOHR
                y *= BOHR
                z *= BOHR
                elem = standardize_elem(parts[3])
                atoms.append((elem, (x, y, z)))
                i += 1
            # consume $end if present
            if i < len(lines) and lines[i].lower().startswith("$end"):
                i += 1
            continue

        i += 1

    if cell_params is None:
        raise ValueError("Did not find a $cell block.")
    if not atoms:
        raise ValueError("Did not find any atoms in $coord block.")

    return alat, atoms


def standardize_elem(elem):
    if len(elem) == 1:
        return elem.upper()
    elif len(elem) == 2:
        return ''.join([elem[0].upper(), elem[1].lower()])
    else:
        raise ValueError


if __name__ == '__main__':
    with open('systems', 'r') as f:
        systems = f.read().splitlines()

    for system in systems:
        print(system)
        alat, atom = parse_structure(f'../{system}')
        print(alat)
        print(atom)

        lat = LAT().init_from_pyscf_atom(atom)
        lat.alat = alat

        out_prefix = system
        lat.dump_all(out_prefix=system)
