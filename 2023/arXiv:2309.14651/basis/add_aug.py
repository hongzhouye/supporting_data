import numpy as np

from pyscf import lib

def load_basis1(fbas, atm, unc=False):
    import os
    from pyscf import gto
    from pyscf.pbc import gto as pbcgto
    if 'ghost' in atm.lower():
        atm = atm.split('-')[1]
    if os.path.isfile(fbas):
        basis = gto.basis.load(fbas, atm)
    else:
        basis = pbcgto.M(atom=f'{atm} 0 0 0', a=np.eye(3)*10, basis=fbas, spin=None)._basis[atm]
    if unc:
        basis = gto.uncontracted_basis(basis)
    return basis


def sort_basis(basis):
    ls_uniq, ls_uniq_inv = np.unique([b[0] for b in basis], return_inverse=True)
    new_basis = []
    for il,l in enumerate(ls_uniq):
        idxs = np.where(ls_uniq_inv==il)[0]
        emaxs = []
        basis_l = []
        for i in idxs:
            ecs = np.asarray(basis[i][1:])
            if ecs.ndim == 1: ecs = ecs.reshape(-1,1)
            es = ecs[:,0]
            if es.size > 1 and not np.all(es[:-1] - es[1:] > 0):
                order = np.argsort(es)[::-1]
                basis_l += [[basis[i][0]]+[basis[i][1+j] for j in order]]
            else:
                basis_l += [basis[i]]
            emaxs.append( es.max() )
        order = np.argsort(emaxs)[::-1]
        new_basis += [basis_l[i] for i in order]
    new_basis = [[int(b[0]), *b[1:]] for b in new_basis]
    return new_basis
def _format_basis(atm, basis, fmt_exp, fmt_coeff, sort):
    LSTR = "SPDFGHIKL"
    def fmt_arr(arr, fmt):
        if '%' in fmt:
            sarr = " ".join([fmt%a for a in arr])
        elif ':' in fmt:
            sarr = " ".join([fmt.format(a) for a in arr])
        else:
            raise RuntimeError('Unknown fmt str')
        return sarr

    if sort:
        basis = sort_basis(basis)

    sout = []
    for b in basis:
        l = int(b[0])
        sout.append("%s  %s" % (atm, LSTR[l]))
        ecs = np.array(b[1:])
        if ecs.ndim == 1:
            ecs = ecs.reshape(1,-1)
        for ec in ecs:
            sec = "   ".join([fmt_arr(ec[:1], fmt_exp),
                              fmt_arr(ec[1:], fmt_coeff)])
            sout.append(sec)
    sout = "\n".join(sout)

    return sout
def format_basis(basis, fmt_exp=r'%12.6f', fmt_coeff=r'% .6e', sort=True):
    ''' Format basis in NWChem format.

    Args:
        basis: PySCF basis dict format. E.g.
            basis = {"H": [[0,(0.7, 1.)]], "F": [[0,(1.5, 1.)],[1,(0.3,1.)]]}
    '''
    assert(isinstance(basis, dict))

    sout = {}
    for atm,bas in basis.items():
        sout[atm] = _format_basis(atm, bas, fmt_exp, fmt_coeff, sort)
    return sout
def dump_basis(basis, fmt_exp=r'%.6f', fmt_coeff=r'% .6e', stdout=None, sort=True):
    ''' Dump basis in NWChem format.

    Args:
        basis: PySCF basis dict format. E.g.
            basis = {"H": [[0,(0.7, 1.)]], "F": [[0,(1.5, 1.)],[1,(0.3,1.)]]}
        stdout: where to dump the output; if not provided --> sys.stdout
    '''
    if stdout is None:
        import sys
        stdout = sys.stdout
    sout = format_basis(basis, fmt_exp, fmt_coeff, sort)
    for atm, s in sout.items():
        stdout.write(s + "\n")
    return sout


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
