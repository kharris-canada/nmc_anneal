# Theory and Physical Model

## Overview

The **nmc_anneal** package models layered nickel–manganese–cobalt (NMC) lithium-ion battery cathodes at the atomistic level. The approach focuses on local charge balance and short-range interactions to reproduce experimentally observed disorder, lithium ordering, and spectroscopic signatures.

---

## Lattice Representation

The chemical structure of layered NMC cathodes consists of alternating 2D sheets of transition metals alternating with 2D sheets of lithium atoms. These metal layers are separated by oxygen sheets. The oxygen sheets do not change in any across different stoichiometries or charge levels, so these are not represented internally. Calculations instead refer to virtual oxygen layers with the same structure for book-keeping purposes.

The structure is represented internally as a stack of 2D arrays. Layer 0 is lithium, layer 1 transition metal, and so on. This stack is stored as a 3D NumPy array with the 0th axis being the stacking axis. To speed up math on the charge checks while also tracking any cases where multiple metals have the same charge, a dual array is used. One with the atomic charges and one with the names of the ions.

Ignoring vacancies and mixtures of TMs, the structure of these lattices is a simple FCC closest packing of spheres (see https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres). In FCC, the layers stack with a repeating ABC, ABC, ABC, etc. order with, with each layer is shifted over 0.5 units along both the a and b directions (∠ a,b = 60 °). The third layer is shifted the same amount/direction. The first three layers therefore look like:

![image](images/Layer_Stacking.png)

The shift from the third layer to the fourth layer is such that the atomic positions overlay exactly with the positions of layer one. However, the code here uses the convention that the fourth layer of the periodic cell is diagonally shifted from layer one. Therefore the bottom left atom of every layer depicted above is at index (L, a=0, b=0). Geometrically, index 0,0 of layer 1 is shifted +1a, +1b from the atom at index 0,0 of layer 0. This choice of unit cell means that every layer in a large stack are connected to the layers above/below in the exact same way, and nearest-neighbor calculations are very easily made periodic with modular arithmetic. While the structure is stored as a 3D Numpy array that looks square, it represents a tilted parallelogram:

![image](images/3Dlattice.png)

The oxygen layers are chemically immutable, so there is no need to store them. However, bookkeeping requires a convention to refer to them for calculating nearest neighbors. Here, we use as a convention that oxygen-layer-0 is between metal-layer-0 (lithium layer) and metal-layer-1 (TM layer); i.e., oxygen layer 0 is physically between the metal layers described by [0,:,:] and [1,:,:] of the NumPy array. This means that every metal layer at vertical index l is sandwiched between oxygen layers with index (l-1) and (l), subject to periodicity when needed.

## Energy Model

The system energy is determined by **local oxygen charge balance**. Each oxygen is coordinated by neighboring cations, and deviations from ideal charge neutrality contribute to the energy (see Harris et al., Chem. Mater. 2017, 29, 5550). 

Key assumptions:

- Only short-range electrostatic effects are included
- Long-range Coulomb interactions are neglected
- Local environments dominate disorder formation

This simplified model allows efficient simulation of large lattices while preserving realistic ordering behavior.


---

## Simulated Annealing

Simulated annealing is used to model disorder formation during synthesis.

Process:

1. Random initial lattice configuration
2. Local swaps between species
3. Energy evaluation using local charge balance
4. Metropolis acceptance rule
5. Temperature gradually lowered

This produces physically realistic cation ordering and clustering.

---

## Delithiation Model

Lithium extraction is simulated using a deterministic local-energy-driven algorithm:

- Lithium removed from highest-energy local environments first
- Generates evolving lithium distributions across the lattice

---

## NMR Model

Synthetic **⁷Li MAS NMR spectra** are generated from local lithium environments.

Each lithium site contributes a resonance based on:

- Neighboring transition metals
- Local charge distribution
- Coordination environment

The final spectrum is constructed by summing contributions from all lithium sites.

---

## Model Limitations

- No long-range electrostatics
- No mixed oxidation states or oxygen participation in electrochemistry
- No local structural distortions
- Not a full DFT model

Despite simplifications, the model reproduces key experimental trends in disorder formation and NMR spectra. Preliminary results of delithiation simulations as studied by 7Li NMR are promising.

