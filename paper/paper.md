---
title: 'hspin: A Python package for first-principles calculation of zero field splitting tensor'
tags:
  - Python
  - density functional theory
  - zero field splitting
  - paramagnetic defects
authors:
  - name: He Ma
    orcid: 1
    affiliation: "1, 2"
  - name: Marco Govoni
    orcid: 1
    affiliation: 3
  - name: Giulia Galli
    orcid: 1
    affiliation: "1, 2, 3"
affiliations:
 - name: Department of Chemistry, University of Chicago, Chicago, IL 60637, USA
   index: 1
 - name: Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA
   index: 2
 - name: aterials Science Division and Center for Molecular Engineering, Argonne National Laboratory, Lemont, IL 60439, USA
   index: 3
date: 06 February 2020
bibliography: paper.bib
---

# Summary
Electron spins in molecules and materials are important resources for information storage and quantum technologies. The low energy physics of electrons spins can be described by spin Hamiltonians, which characterize the interactions between electron spins and between electrons spins and external environment. For systems with effective electron spin $S \geq 1$ at zero magnetic and hyperfine field, the spin Hamiltonian assumes the form
$$ H_\mathrm{spin} = \mathbf{S} \cdot \mathbf{D} \cdot \mathbf{S} $$
where $\mathbf{S}$ is the spin operator and $\mathbf{D}$ denotes the zero field splitting (ZFS) tensor. Physically, the ZFS tensor originates from magnetic spin-spin interactions between electrons and spin-orbit interactions. For systems without heavy elements, in particular main group spin defects in semiconductors, the spin-spin component is dominant. The ZFS tensor determines the energy splitting of spin sublevels without external fields and is an important property for open-shell molecules and spin defects in semiconductors with $S \geq 1$.

`hspin` is an MPI-parallelized Python code for the first-principles calculation of spin-spin ZFS tensor based on wavefunctions obtained from density functional theory (DFT) calculations. The spin-spin ZFS tensor can be represented as an expectation value of dipole-dipole interactions on the DFT Kohn-Sham orbitals
$$ D_{ab} = \frac{1}{2} \frac{1}{S(2S-1)} \frac{\mu_0}{4\pi} (\gamma_\text{e} \hbar)^2 \ \sum_{i \le j}^{\text{occ.}} \chi_{ij} \langle \Phi_{ij}| \  \frac{ r^2\delta_{ab} - 3r_a r_b }{ r^5 } \  |\Phi_{ij} \rangle $$
where $a, b = x, y, z$ are Cartesian indices; $\gamma_e$ is the gyromagnetic ratio of electron; the summation is over all pairs of occupied Kohn-Sham orbitals; $\chi_{ij} = \pm 1$ for parallel and antiparallel spins respectively; $\Phi_{ij}(\textbf{r},\textbf{r}')$ are $2 \times 2$ determinants formed from orbitals $\phi_{i}$ and $\phi_{j}$, $\Phi_{ij}(\textbf{r},\textbf{r}')=\frac{1}{\sqrt{2}}\Big[\phi_{i}(\textbf{r})\phi_{j}(\textbf{r}') - \phi_{i}(\textbf{r}')\phi_{j}(\textbf{r})]$. `hspin` implements the numerical formalism proposed in [@Rayson:2008] to evaluate the above expression on plane-wave basis. The real space integrations are transformed into summations over reciprocal lattice vectors, and the transformation is facilitated by Fast Fourier Transform (FFT). The summation over all pairs of Kohn-Sham orbitals is distributed into a square grid of processors through MPI, which significantly reduce the CPU time and memory cost per processor.

`hspin` can work with output files of various plane-wave DFT codes. For instance, `hspin` can directly read DFT wavefunctions from `Quantum Espresso` [@Giannozzi:2009] in the HDF5 format; `hspin` also supports the standard cube file format, which allows it to work with other DFT codes such as `Qbox` [@Gygi:2008].

Since its development, `hspin` has been adopted by several works to predict ZFS tensors for spin defects in semiconductors, and facilitated exciting progresses in the discovery of novel spin defects [@Seo:2017] and the coherence control of defect spins in crystals [@Whiteley:2019]. Thanks to the parallel design of the code, `hspin` can perform calculations for defects embedded in large supercells. The calculations performed in [@Whiteley:2019] used supercells that contain more than 3000 valence electrons, and are among the largest first-principles calculations of ZFS tensors reported by the time this document is written.

# Acknowledgements
We thank Hosung Seo for helpful discussions. 
This work was supported by MICCoM, as part of the Computational Materials Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division through Argonne National Laboratory, under contract number DE-AC02-06CH11357.
This research used computational resources of the University of Chicago Research Computing Center.

# References

