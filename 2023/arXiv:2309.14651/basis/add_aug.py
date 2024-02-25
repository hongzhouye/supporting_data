import numpy as np

from pyscf import lib

from zflow.pyscf_helper import load_basis1, dump_basis


def add_aug(atm, fbas, atm_ref, fbas_ref):
    from pyscf import gto
    bas_ref = load_basis1(fbas_ref, atm_ref)
    bas = load_basis1(fbas, atm)

    def get_bas_data(basis, atm):
        mol = gto.M(atom=atm, basis=basis, spin=None)
        ls = np.asarray([mol.bas_angular(i) for i in range(mol.nbas)])
        es = np.asarray([mol.bas_exp(i).min() for i in range(mol.nbas)])
        return ls, es

    ls_ref, es_ref = get_bas_data(bas_ref, atm_ref)
    ls, es = get_bas_data(bas, atm)

    rats = []
    bas_add = []
    for l in range(lib.param.L_MAX+1):
        ids = np.where(ls==l)[0]
        if ids.size == 0:
            break
        ids_ref = np.where(ls_ref==l)[0]
        es_ref_l = np.sort(es_ref[ids_ref])
        r_ref = es_ref_l[1] / es_ref_l[0]
        e_add = np.min(es[ids]) / r_ref
        bas_add.append( [l, (e_add, 1.)] )

    bas_aug = bas + bas_add
    return bas_aug


if __name__ == '__main__':
    # atms = ['H','He',
    #         'Li','Be','B','C','N','O','F','Ne',
    #         'Na','Mg','Al','Si','P','S','Cl','Ar']
    atms = ['Mg','C','O']

    zetas = ['dz','tz','qz']
    for zeta in zetas:
        fbas_ref = f'aug-cc-pv{zeta}'
        fbas = f'cc-pv{zeta}.dat'
        fout = f'aug-cc-pv{zeta}.dat'

        with open(fout, 'w') as f:
            for atm in atms:
                atm_ref = atm
                basis_aug = add_aug(atm, fbas, atm_ref, fbas_ref)
                f.write('#BASIS SET:\n')
                dump_basis({atm:basis_aug}, stdout=f)


        fbas_ref = f'aug-cc-pv{zeta}-jkfit'
        fbas = f'cc-pv{zeta}-jkfit.dat'
        fout = f'aug-cc-pv{zeta}-jkfit.dat'

        with open(fout, 'w') as f:
            for atm in atms:
                if atm == 'Mg':
                    atm_ref = 'Al'
                else:
                    atm_ref = atm
                basis_aug = add_aug(atm, fbas, atm_ref, fbas_ref)
                f.write('#BASIS SET:\n')
                dump_basis({atm:basis_aug}, stdout=f)
