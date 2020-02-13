---
title: 'hspin: A Python package for first-principles calculations of zero-field splitting tensors'
tags:
  - materials science
  - electron spin
  - zero field splitting
  - density functional theory
  - point defects
authors:
  - name: He Ma
    orcid: 1
    affiliation: "1, 2"
  - name: Marco Govoni
    orcid: 0000-0001-6303-2403
    affiliation: "3, 2"
  - name: Giulia Galli
    orcid: 1
    affiliation: "1, 2, 3"
affiliations:
 - name: Department of Chemistry, University of Chicago, Chicago, IL 60637, USA
   index: 1
 - name: Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA
   index: 2
 - name: Materials Science Division and Center for Molecular Engineering, Argonne National Laboratory, Lemont, IL 60439, USA
   index: 3
date: 06 February 2020
bibliography: paper.bib
---

# Summary
Electron spins in molecules and materials are important resources for information storage and quantum technologies. In order to understand the physical properties of electron spins, one needs to describe the interaction of electron spins in the presence of external electromagnetic fields. Such a description may be achieved by using spin Hamiltonians, with parameters derived from experiments or from calculations. For systems with a single effective electron spin, the leading terms in the spin Hamiltonian are 
$$ H=\mu_B \mathbf{B}\cdot\mathbf{g}\cdot\mathbf{S} + \mathbf{S} \cdot \mathbf{D} \cdot \mathbf{S}$$ 
where $\mu_B is the Bohr magneton, $\mathbf{S}$ is the effective electron spin, $\mathbf{B}$ is the external magnetic field, $\mathbf{g}$ and $\mathbf{D}$ are rank-2 tensors that characterize the strength of electron Zeeman interaction, and zero-field splitting (ZFS). The spin Hamiltonian parameters $\mathbf{g}$ and $\mathbf{D}$ may be obtained by electron paramagnetic resonance (EPR). The ZFS tensor yields the energy splitting of spin sublevels without external fields and is an important property for open-shell molecules and spin defects in semiconductors with $S \geq 1$. Theoretically it can be determined by first-principles electronic structure calculations, which also provide important information complementary to experiments. In the case of spin defects in solids often times the atomistic structure and charge state of the defect are not straightforward to determine, experimentally. Comparing the computed spin Hamiltonian parameters for candidate structures and charge states with experimental results is a useful means to identify the properties of the defect. In addition, first-principles calculations can provide insights into the structure-property relations of molecules and spin defects, thus facilitating the rational design of molecules and materialswith desirable spin properties. Therefore, in order to devise predictive computational strategies, the development of robust methods for the calculation of spin Hamiltonian parameters is an important task. 

In this work we describe the code `hspin` for the calculation of zero-field splitting (ZFS) tensor $\mathbf{D}$ based on wavefunctions obtained from density functional theory (DFT) calculations. For systems without heavy elements, i.e. where spin-orbit interactions are negligible, the ZFS is determined by spin-spin interactions, and can be represented as an expectation value of dipole-dipole interactions on the DFT Kohn-Sham orbitals 
$$ D_{ab} = \frac{1}{2} \frac{1}{S(2S-1)} \frac{\mu_0}{4\pi} (\gamma_\text{e} \hbar)^2 \ \sum_{i \le j}^{\text{occ.}} \chi_{ij} \langle \Phi_{ij}| \  \frac{ r^2\delta_{ab} - 3r_a r_b }{ r^5 } \  |\Phi_{ij} \rangle $$
where $a, b = x, y, z$ are Cartesian indices; $\gamma_e$ is the gyromagnetic ratio of electron; the summation is taken over all pairs of occupied Kohn-Sham orbitals; $\chi_{ij} = \pm 1$ for parallel and antiparallel spins respectively; $\Phi_{ij}(\textbf{r},\textbf{r}')$ are $2 \times 2$ determinants formed from orbitals $\phi_{i}$ and $\phi_{j}$, $\Phi_{ij}(\textbf{r},\textbf{r}')=\frac{1}{\sqrt{2}}\Big[\phi_{i}(\textbf{r})\phi_{j}(\textbf{r}') - \phi_{i}(\textbf{r}')\phi_{j}(\textbf{r})]$. `hspin` adopts the numerical formalism proposed in [@Rayson:2008] to evaluate the above expression on plane-wave basis, which utilize Fourier Transform to reduce the double integrations in real space into a single summation over reciprocal lattice vectors. 

We note that large-scale DFT calculations can yield wavefunction files larger than 50 GB or even 100 GB. Therefore, proper distribution and management of data is critical. In `hspin`, the summation over pairs of Kohn-Sham orbitals is distributed into a square grid of processors, which significantly reduces the CPU time and memory cost per processor. Processors communicate through the Message Passing Interface (MPI) [REF]. `hspin` also implements different memory management modes, which control whether intermediate quantities are kept in memory for reuse or recomputed every time, allowing the user to balance the computational time and memory cost.

`hspin` can work with wavefunctions generated by various plane-wave DFT codes. For instance, `hspin` can directly read DFT wavefunctions from `Quantum Espresso` [@Giannozzi:2009] in the HDF5 format and wavefunction from `Qbox` [@Gygi:2008] in the XML format. The standard cube file format is also supported [REF]. `hspin` features a modular design and utilizes abstract classes for extensibility. Support for new wavefunction format may be easily implemented by defining subclasses of the relevant abstract class and overriding corresponding abstract methods.

Since its development, `hspin` has been adopted by several works to predict ZFS tensors for spin defects in semiconductors, and facilitated exciting progresses in the discovery of novel spin defects [@Seo:2017], and the coherence control of defect spins in crystals [@Whiteley:2019]. `hspin` has also been adopted to generate benchmark data for the development of methods to compute the ZFS tensor with basis sets alternative to plane waves [@Ghosh:2019]. Thanks to the parallel design of the code, `hspin` can perform calculations for defects embedded in large supercells. The calculations performed in [@Whiteley:2019] used supercells that contain more than 3000 valence electrons, and are among the largest first-principles calculations of ZFS tensors reported by the time this document is written. 

# Acknowledgements
We thank Hosung Seo for helpful discussions. This work was supported by MICCoM, as part of the Computational Materials Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division through Argonne National Laboratory, under contract number DE-AC02-06CH11357.

# References

