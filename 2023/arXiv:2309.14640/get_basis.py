import numpy as np

from zflow.pyscf_helper import load_basis1, dump_basis

from pyscf import gto


if __name__ == '__main__':
    def get_basis(atm, zeta):
        if atm == 'Ti':
            from cc_basis_data import get_basis as gbas
            b = gbas(atm, f'cc-pv{zeta}')
            b = gto.M(atom=f'{atm} 0 0 0', basis=b, spin=None)._basis[atm]
        else:
            fbas = f'/Users/hzye/local/opt/ccgto4s/basis/gth-hf-rev/cc-pv{zeta}-lc.dat'
            b = load_basis1(fbas, atm)
        return b

    atms = ['H','O','Al','Ti']
    for zeta in ['dz','tz','qz']:
        fout = f'gth-cc-pv{zeta}.dat'
        with open(fout, 'w') as f:
            for atm in atms:
                b = get_basis(atm, zeta)
                print(atm, len(b))
                f.write('#BASIS SET:\n')
                dump_basis({atm:b}, stdout=f)
