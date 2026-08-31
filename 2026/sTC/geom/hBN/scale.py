import sys
import numpy as np

from zflow.pyscf_helper import LAT


if __name__ == '__main__':
    f0 = '1.000.vasp'
    lat0 = LAT().init_from_vasp_poscar(f0)
    atms = lat0.atms
    rs = lat0.rs
    alat = lat0.alat

    for scale in [0.965, 0.974, 0.983, 0.992, 1.000, 1.008, 1.016, 1.024, 1.032]:
        out_prefix = f'{scale:.3f}'

        atom = [(atm,r) for atm,r in zip(atms,rs*scale)]
        lat = LAT().init_from_pyscf_atom(atom)
        lat.alat = alat*scale

        lat.dump_all(out_prefix)
