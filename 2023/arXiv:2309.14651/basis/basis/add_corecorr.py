import numpy as np
import shutil

from pyscf import lib

from zflow.pyscf_helper import load_basis1, dump_basis


def add_core(atm, fbas, atm_ref, fbas_ref):
    from pyscf import gto
    if fbas_ref is None:
        bas_ref = []
    else:
        bas_ref = load_basis1(fbas_ref, atm_ref)
    bas = load_basis1(fbas, atm)

    bas_core = bas_ref + bas

    return bas_core


if __name__ == '__main__':
    # atms = ['H','He',
    #         'Li','Be','B','C','N','O','F','Ne',
    #         'Na','Mg','Al','Si','P','S','Cl','Ar']
    atms = ['Mg','C','O']

    zetas = ['dz','tz','qz']
    for zeta in zetas:
        ZETA = zeta.upper()
        # cc-pvxz --> cc-pcvxz
        fbas = f'cc-pv{zeta}.dat'
        fout = f'cc-pcv{zeta}.dat'

        with open(fout, 'w') as f:
            for atm in atms:
                if atm in ['Mg']:
                    fbas_ref = f'/Users/hzye/local/opt/pyscf/pyscf/gto/basis/cc-pCV{ZETA}.dat'
                else:
                    fbas_ref = None
                atm_ref = atm
                basis_core = add_core(atm, fbas, atm_ref, fbas_ref)
                f.write('#BASIS SET:\n')
                dump_basis({atm:basis_core}, stdout=f)

        # aug-cc-pvxz --> aug-cc-pcvxz
        fbas = f'aug-cc-pv{zeta}.dat'
        fout = f'aug-cc-pcv{zeta}.dat'

        with open(fout, 'w') as f:
            for atm in atms:
                if atms in ['Mg']:
                    fbas_ref = f'/Users/hzye/local/opt/pyscf/pyscf/gto/basis/cc-pCV{ZETA}.dat'
                else:
                    fbas_ref = None
                atm_ref = atm
                basis_core = add_core(atm, fbas, atm_ref, fbas_ref)
                f.write('#BASIS SET:\n')
                dump_basis({atm:basis_core}, stdout=f)

        # cc-pvxz-jkfit --> cc-pcvxz-jkfit
        fbas = f'cc-pv{zeta}-jkfit.dat'
        fout = f'cc-pcv{zeta}-jkfit.dat'
        shutil.copyfile(fbas, fout)

        # aug-cc-pvxz-jkfit --> aug-cc-pcvxz-jkfit
        fbas = f'aug-cc-pv{zeta}-jkfit.dat'
        fout = f'aug-cc-pcv{zeta}-jkfit.dat'
        shutil.copyfile(fbas, fout)
