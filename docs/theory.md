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

Lithium extraction is simulated using a deterministic local-energy-driven algorithm using the neighboring oxygen atoms.

- Lithium removed from highest-energy local environments first
- The next move when degenerate energies occur are randomly selected
- This oxidation/delithiation generates evolving lithium distributions across the lattice

Three models for the oxidation steps are provided for the user to test:
* Ni<sup>2+</sup> directly to >Ni<sup>4+</sup> (and then followed by Co<sup>3+</sup> if at extremely high capacity). Oxidation model = "ni_2to4_co_3to4"
* Ni<sup>2+</sup> to Ni<sup>3+</sup>, and then once complete, Ni<sup>3+</sup> directly to Ni<sup>4+</sup> begins (and Co<sup>3+</sup> if needed). Oxidation model = "ni_2to3_ni_3to4_co_3to4"
* Ni<sup>2+</sup> to Ni<sup>3+</sup>, and then once complete, EITHER Ni<sup>3+</sup> directly to Ni<sup>4+</sup> or Co<sup>3+</sup> to Co<sup>4+</sup> according to oxygen energy at each atomic step. Oxidation model = "ni_2to3_any_3to4"

---

## NMR Model

Synthetic **⁷Li MAS NMR spectra** are generated from local lithium environments. These are determined by the identity and geometric connection to nearest-neighbor TM atoms (see Harris et al., Chem. Mater. 2017, 29, 5550; Zeng et al.,  Chem. Mater. 2007, 19, 6277; Grey et al., Chem. Rev. 2004, 104, 4493 and references therein). The current version of the code assumes any Li atoms you place in the nominally TM layer behave the same as those in the nominally Li layer, which needs experimental data and can then be improved. It is best to consider these shifts as fitting parameters within a small range because they are certainly temperature dependent (and thereby MAS-rate dependent), and may differ between samples.

Each lithium site contributes a single resonance based on:

- \# and identity of TMs at 90 bond angles in neighboring layers
- \# and identity of TMs at 180 bond angles in neighboring layers
- \# and identity of TMs at 90 bond angles in the same layer (it is more common to not have Li and TM in the same layer, see Li2MnO3 literature for a well known case to test on though)

The final spectrum is constructed by summing contributions from all lithium sites.

---

## Model Limitations

- No long-range electrostatics (particularly TM-layer to TM-layer forces which order Li2MnO3 and variants, but are clearly quite weak and may not affect quenched samples)
- No mixed oxidation states or oxygen participation in electrochemistry
- No local structural distortions
- Not a full DFT model

Despite simplifications, the model reproduces key experimental trends in disorder formation and NMR spectra. Preliminary results of delithiation simulations as studied by 7Li NMR are promising.

