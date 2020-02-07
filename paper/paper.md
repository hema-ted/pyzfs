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
Electron spins in molecules and materials are important resources for information storage and quantum information technology. The low energy physics of spins can be described by spin Hamiltonians, which characterize the interactions between electron spins and potentially also with environment. The zero field splitting (ZFS) tensor is an important parameter in spin Hamiltonians for paramagnetic molecules and defects in solids with electron spins $S \leq 1$. The ZFS tensor and originates from magnetic spin-spin interactions between electrons as well as spin-orbit interactions. ZFS tensor determines the energy splitting of spin sublevels without external magnetic field, and is a crucial descriptor for novel molecular magnets and quantum sensing materials. 

$$ H_{ZFS} = S D S $$

`hspin` is a MPI-parallelized Python code for first-principles calculation of spin-spin ZFS tensor based on wavefunctions obtained from density functional theory (DFT) calculations. `hspin` mainly focus on the calculation of ZFS tensor for condensed systems such as defects in semiconductors, which are commonly studied with plane-wave pseudopotential DFT. 

$$ D_{ab} = \frac{1}{2} \frac{1}{S(2S-1)} \frac{\mu_0}{4\pi} (\gamma_\text{e} \hbar)^2 \ \sum_{i \le j}^{\text{occ.}} \chi_{ij} \langle \Psi_{ij}| \  \frac{ r^2\delta_{ab} - 3r_a r_b }{ r^5 } \  |\Psi_{ij} \rangle $$


`hspin` implements the numerical formalism in [@Rayson:2008] for the evaluation of ZFS in the reciprocal space using fast fourier transform (FFT). 

$$ I_{ab} = 4\pi \Omega \sum_{\mathbf{G} \neq 0} { \rho(\mathbf{G}, -\mathbf{G}) \left( \frac{G_a G_b}{G^2} - \frac{\delta_{ab}}{3} \right)
 } $$

`hspin` can parse the output files of various plane-wave DFT codes. For instance, `hspin` can directly read DFT wavefunctions from `Quantum Espresso` [@Giannozzi:2009] in the HDF5 format; `hspin` also supports the standard cube file format and thus work with any DFT codes that can output cube files, such as `Qbox` [@Gygi:2008]. 

Since its development, `hspin` has been adopted by several works to predict ZFS tensors for spin defects in semiconductors, and facilitated exciting research in the discovery of novel defects [@Seo:2017] and the coherence control of defect electron spin in crystals [@Whiteley:2019]. Thanks to the parallel design of the code, `hspin` can perform large-scale calculations for defects in large supercells. The supercells used in [@Whiteley:2019] contain more than 2000 valence electrons, which are among the largest first-principles calculations of ZFS tensors reported so far.

# Acknowledgements
We thank Hosung Seo for helpful discussions. 
This work was supported by MICCoM, as part of the Computational Materials Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division through Argonne National Laboratory, under contract number DE-AC02-06CH11357.
This research used computational resources of the University of Chicago Research Computing Center.

# References

