# Data for "Smoothed truncated Coulomb potential for periodic Gaussian-basis Hartree--Fock exchange"

## Overview

This directory contains the numerical data underlying the main-text and Supporting Information results for the accompanying manuscript. The data were generated from periodic Hartree--Fock (HF) calculations using the Wigner--Seitz smoothed truncated Coulomb (WS-sTC), Wigner--Seitz truncated Coulomb (WS-TC), and probe-charge Ewald treatments of finite-size effects.

Most files are comma-separated text files with a header row. Files in the `gapped/`, `metal/`, `eos/`, and `eos_fit/` directories use CSV formatting even when they have no `.csv` extension. Tabular energy columns are reported in Hartree.

## Directory structure

| Directory | Contents |
|---|---|
| `gapped/gth/` | HF exchange energies versus `k`-point mesh for diamond, silicon, and h-BN using GTH pseudopotentials. |
| `gapped/alle/` | Corresponding all-electron HF exchange energies. |
| `metal/gth/` | HF exchange energies versus `k`-point mesh for lithium and aluminum using GTH pseudopotentials. |
| `metal/alle/` | Corresponding all-electron HF exchange energies. |
| `eos/gth/` | Diamond HF energy--volume data using GTH pseudopotentials. |
| `eos/alle/` | Corresponding all-electron diamond energy--volume data. |
| `eos_fit/gth/` | Equilibrium lattice constants and bulk moduli fitted from the GTH-pseudopotential equation-of-state data. |
| `eos_fit/alle/` | Corresponding fits to the all-electron equation-of-state data. |
| `bands/gth/` | PySCF checkpoint files containing diamond and silicon band structures, together with a loading and plotting script. |
| `molcrys/CO2/` | Per-molecule HF energies and smoothing parameters for crystalline CO2. |
| `molcrys/NH3/` | Per-molecule HF energies and smoothing parameters for crystalline NH3. |
| `molcrys/TTF/` | Per-molecule HF energies and smoothing parameters for the tetrathiafulvalene (TTF) crystal. |

Here, `gth` denotes GTH-pseudopotential calculations and `alle` denotes all-electron calculations.

The directories map to the manuscript as follows: `gapped/` supports Fig. 2 and Figs. S1--S3; `eos/` and `eos_fit/` support Fig. 3 and Table I; `bands/` supports Fig. 4 and S4 and Table II; `metal/` supports Fig. 5 and Figs. S3 and S5; and `molcrys/` supports Figs. 6--7.

## File naming

- System labels: `C` = diamond, `Si` = silicon, `hBN` = hexagonal boron nitride, `Li` = lithium, and `Al` = aluminum.
- Basis labels: `sz`, `dz`, and `tz` denote single-, double-, and triple-zeta basis sets, respectively; `minao` denotes the MINAO basis.
- `ewald` denotes the probe-charge Ewald treatment.
- `vcut_ws` denotes WS-TC.
- `vcut_ws+N` and `stc_vcut_ws+N` denote WS-sTC with eta = `N`, where `N` is 2, 3, 4, or 5.
- A three-digit mesh label `abc` denotes an `a x b x c` `k`-point mesh; for example, `444` denotes a `4 x 4 x 4` mesh and `331` denotes a `3 x 3 x 1` mesh.
- In `eos/`, the mesh label is included in the filename. In `eos_fit/`, the fitted values for different meshes are stored as separate rows.
- Files ending in `_omega.csv` contain the value of the smoothing parameter omega associated with each `k`-point mesh.

## Columns and units

| Column | Description | Unit |
|---|---|---|
| `kmesh` | Compact `k`-point mesh label, as defined above | dimensionless |
| `ek` | HF exchange energy | Hartree |
| `ehf` | HF total energy | Hartree |
| `etot` | HF total energy used for the equation-of-state fit | Hartree |
| `vol` | Volume of the two-atom diamond primitive cell | Angstrom^3 |
| `omega` | Range-separation/smoothing parameter omega | Bohr^-1 |
| `a0` | Fitted equilibrium lattice constant | Angstrom |
| `B0` | Fitted bulk modulus | GPa |

## Band-structure files

The `bands/gth/*.chk` files are HDF5-format PySCF checkpoint files. They contain the band energies and associated path metadata under the keys used by `bands/plot_bands_paper.py`, including `bands`, `kpath`, `latvec`, `points`, `sp_points`, and `sp_points_name`.

The example script can be run from the `bands/` directory:

```bash
python plot_bands_paper.py
```

Edit the system-selection variables near the beginning of the script to choose diamond or silicon. The script requires Python with NumPy, Matplotlib, and PySCF; it loads the checkpoint data, prints the direct and indirect gaps, and writes a PDF band-structure plot. The script interprets the stored band energies in eV.

## Citation

- Zendo: 10.5281/zenodo.22212380
