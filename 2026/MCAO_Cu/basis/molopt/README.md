# Cu TZV2P-MOLOPT basis and generated auxiliary basis

This folder contains the Cu `wB97X-V-GTH-q19` TZV2P-MOLOPT orbital
basis used in the Cu basis-set calculations, together with the PySCF
density-fitting auxiliary basis generated for that orbital basis.

## Files

- `tzv2p.dat`: orbital TZV2P-MOLOPT basis
- `tzv2p-fit.dat`: generated density-fitting auxiliary basis

## Reference for the TZV2P-MOLOPT orbital basis

The orbital basis is the Cu `TZV2P-MOLOPT-wB97X-V-GTH-q19` basis from:

Wan-Lu Li, Kaixuan Chen, Elliot Rossomme, Martin Head-Gordon, and
Teresa Head-Gordon, "Optimized Pseudopotentials and Basis Sets for
Semiempirical Density Functional Theory for Electrocatalysis
Applications," J. Phys. Chem. Lett. 2021, 12, 10304-10309.
DOI: 10.1021/acs.jpclett.1c02918

For the general MOLOPT basis-set construction strategy, see:

Joost VandeVondele and Juerg Hutter, "Gaussian Basis Sets for Accurate
Calculations on Molecular Systems in Gas and Condensed Phases,"
J. Chem. Phys. 2007, 127, 114105. DOI: 10.1063/1.2770708

## How the auxiliary basis was optimized

The auxiliary basis `tzv2p-fit.dat` was generated in this work for PySCF
density fitting / RPA calculations. 

- orbital basis: `wB97X-V-GTH-q19/tzv2p`
- pseudopotential: `gth-pbe-q19`
- atomic reference: Cu atom
- target cost: `1e-5`
- `fac_vjvk`: `2.0`

