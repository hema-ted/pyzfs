---
title: 'PyZFS: A Python package for first-principles calculations of zero-field splitting tensors'
tags:
  - materials science
  - electron spin
  - zero field splitting
  - density functional theory
  - point defects
authors:
  - name: He Ma
    orcid: 0000-0001-8987-8562
    affiliation: "1,3"
  - name: Marco Govoni
    orcid: 0000-0001-6303-2403
    affiliation: "3,2"
  - name: Giulia Galli
    orcid: 0000-0002-8001-5290
    affiliation: "1,2,3"
affiliations:
 - name: Department of Chemistry, University of Chicago, Chicago, IL 60637, USA
   index: 1
 - name: Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA
   index: 2
 - name: Materials Science Division and Center for Molecular Engineering, Argonne National Laboratory, Lemont, IL 60439, USA
   index: 3
date: 28 February 2020
bibliography: paper.bib
---

# Summary
Electron spins in molecules and materials may be manipulated and used to store information, and hence they are interesting resources for quantum technologies.
A way to understand the physical properties of electron spins is to probe their interactions with electromagnetic fields.
Such interactions can be described by using a so-called spin Hamiltonian, with parameters derived from either experiments or calculations.
For a single electron spin (e.g., associated to a point-defect in a semiconductor or insulator), the leading terms in the spin Hamiltonian are
$$ H=\mu_B \mathbf{B}\cdot\mathbf{g}\cdot\mathbf{S} + \mathbf{S} \cdot \mathbf{D} \cdot \mathbf{S} $$
where $\mu_B$ is the Bohr magneton, $\mathbf{S}$ is the electron spin operator, $\mathbf{B}$ is an external magnetic field, $\mathbf{g}$ and $\mathbf{D}$ are rank-2 tensors that characterize the strength of the Zeeman interaction, and the zero-field splitting (ZFS), respectively.
Experimentally, the spin Hamiltonian parameters $\mathbf{g}$ and $\mathbf{D}$ may be obtained by electron paramagnetic resonance (EPR).
The ZFS tensor describes the lifting of degeneracy of spin sublevels in the absence of external magnetic fields, and is an important property of open-shell molecules and spin defects in semiconductors with spin quantum number $S \geq 1$.
The ZFS tensor can be predicted from first-principles calculations, thus complementing experiments and providing valuable insight into the design of novel molecules and materials with desired spin properties.
Furthermore, the comparison of computed and measured ZFS tensors may provide important information on the atomistic structure and charge state of defects in solids, thus helping to identify the defect configuration present in experimental samples.
Therefore, the development of robust methods for the calculation of the ZFS tensor is an interesting topic in molecular chemistry and materials science.

In this work we describe the code `PyZFS` for the calculation of the ZFS tensor $\mathbf{D}$ of molecules and solids, based on wavefunctions obtained from density functional theory (DFT) calculations.
For systems without heavy elements, i.e., where spin-orbit coupling is negligible, magnetic spin-spin interactions are the dominant ones in the determination of the ZFS tensor.
For molecules and materials with magnetic permeability close to the vacuum permeability $\mu_0$, the spin-spin ZFS tensor evaluated using the DFT Kohn-Sham wavefunctions, is given by @Harriman:1978:
$$ D_{ab} = \frac{1}{2S(2S-1)} \frac{\mu_0}{4\pi} (\gamma_\mathrm{e} \hbar)^2 \ \sum_{i < j}^{\mathrm{occ.}} \chi_{ij} \langle \Phi_{ij}| \ \frac{ r^2\delta_{ab} - 3r_a r_b }{ r^5 } \ |\Phi_{ij} \rangle $$
where $a, b = x, y, z$ are Cartesian indices; $\gamma_e$ is the gyromagnetic ratio of electrons; the summation runs over all pairs of occupied Kohn-Sham orbitals; $\chi_{ij} = \pm 1$ for parallel and antiparallel spins, respectively; $\Phi_{ij}(\textbf{r},\textbf{r}')$ are $2 \times 2$ determinants formed from Kohn-Sham orbitals $\phi_{i}$ and $\phi_{j}$, $\Phi_{ij}(\textbf{r},\textbf{r}')=\frac{1}{\sqrt{2}}\Big[\phi_{i}(\textbf{r})\phi_{j}(\textbf{r}') - \phi_{i}(\textbf{r}')\phi_{j}(\textbf{r})]$.

Several quantum chemistry codes (for example, ORCA [@Neese:2012]) include the implementation of ZFS tensor calculations for molecules, where electronic wavefunctions are represented using Gaussian basis sets.
However, few open-source codes are available to compute ZFS tensors using plane-wave basis sets, which are usually the basis sets of choice to study condensed systems.
In `PyZFS` we implement the evaluation of spin-spin ZFS tensors using plane-wave basis sets.
The double integration in real space is reduced to a single summation over reciprocal lattice vectors through the use of Fast Fourier Transforms [@Rayson:2008].

We note that a large-scale DFT calculations can yield wavefunction files occupying tens of gigbytes.
Therefore, proper distribution and management of data is critical.
In `PyZFS`, the summation over pairs of Kohn-Sham orbitals is distributed into a square grid of processors through the use of the Message Passing Interface (MPI), which significantly reduces the CPU time and memory cost per processor.

`PyZFS` can use wavefunctions generated by various plane-wave DFT codes as input.
For instance, it can directly read wavefunctions from Quantum Espresso [@Giannozzi:2009] in the HDF5 format and from Qbox [@Gygi:2008] in the XML format.
The standard cube file format is also supported.
`PyZFS` features a modular design and utilizes abstract classes for extensibility.
Support for new wavefunction format may be easily implemented by defining subclasses of the relevant abstract class and overriding corresponding abstract methods.

Since its development, `PyZFS` has been adopted to predict ZFS tensors for spin defects in semiconductors, and facilitated the discovery of novel spin defects [@Seo:2017] and the study of spin-phonon interactions in solids [@Whiteley:2019].
`PyZFS` has also been adopted to generate benchmark data for the development of methods to compute the ZFS tensor using all electron calculations on finite element basis sets [@Ghosh:2019].
Thanks to the parallel design of the code, `PyZFS` can perform calculations for defects embedded in large supercells.
For example, the calculations performed by @Whiteley:2019 used supercells that contain more than 3000 valence electrons, and are among the largest first-principles calculations of ZFS tensors reported at the time this document was written.

# Acknowledgements
We thank Hosung Seo and Viktor Ivády for helpful discussions.
We thank Malcolm Ramsay and Xiao Wang for reviewing this paper and providing helpful comments and suggestions.
This work was supported by MICCoM, as part of the Computational Materials Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division through Argonne National Laboratory, under contract number DE-AC02-06CH11357.

# References

