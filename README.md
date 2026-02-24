# nmc_anneal

# nmc_anneal

**Physics-informed lattice annealing for modeling cation disorder, phase behavior, and electrochemical states in layered oxide materials.**

---

## Why this package exists

Layered transition-metal oxides of the NMC (layered Ni/Mn/Co cathodes) are central to modern electrochemical energy storage, yet **predicting their microscopic cation arrangement and phase behavior remains difficult** since these materials are all created with intrinsic disorder via high temperature synthesis followed by a quench. The often hundreds of distinct local cation arrangments are impossible to to directly extract from experimental techniques (XRD, NMR, PXRD, electrochemistry) without good proposed structures given that the experiments provide only indirect, averaged, or summed information. Such structures are too complicated for ab initio molecular dynamics calculations in realistic timeframes.

**nmc_anneal was created to bridge this gap.**

This package provides a fast, reproducible, and physically motivated lattice annealing framework that allows researchers to:

* Generate realistic transition-metal configurations
* Study ordering, disorder, and clustering
* Explore phase behavior across composition and temperature
* Connect microscopic structure to measurable observables
* Produce datasets suitable for machine learning and statistical analysis
* Connect starting structures to ab initio property calculations

The nmc_anneal package provides an extension of the 2D methods we published in *Chem. Mater.* 29, pp 5550, **2017**. Here, 3D structures and algorithms are built to treat more complicated members of the NMC family. Even more importantly, they allow for the study of charging/discharging the cathode.

Rather than replacing first-principles methods, **nmc_anneal complements them** by enabling large-scale configurational exploration guided by physically meaningful energy models. Understanding the electrochemical performance of distinct metal arrangments and deteriming how to select for them would produce higher capacity batteries for use in many industries.

---

## Core idea

The package simulates a discrete lattice of species (e.g., Ni, Mn, Co, Li,  vacancies, oxidation states) and evolves it through **Monte-Carlo-style annealing** under a configurable energy model. This allows realistic disordered and partially ordered states to emerge naturally rather than being imposed.

Key design goals:

* **Fast enough for large parameter sweeps**
* **Simple enough to modify**
* **Physics-motivated, not purely abstract**
* **Reproducible and testable**
* **Usable for both research and data generation**

---

## What makes this different

Unlike many lattice or Monte Carlo codes, this package is purpose-built to capture the unique physics of these layered oxide cathodes. It emphasizes:

### 1. Research workflow integration

Designed to work naturally with:

* Experimental comparisons
* Phase diagram generation
* Convergence studies
* Visualization tools
* Dataset generation for ML

### 2. Configurable physics

Users can define:

* Interaction models
* Species and oxidation states
* Temperature schedules
* Lattice size and geometry
* Annealing protocols

### 3. Reproducible simulation objects

Simulations are controlled through structured configuration objects, making runs:

* Reproducible
* Serializable
* Easy to batch
* Easy to analyze

### 4. Built for exploration, not just single runs

The package includes tools for:

* Phase diagram mapping
* Convergence detection
* Automated sweeps
* Visualization of lattice states

---

## Typical use cases

* Predicting local cation order at any stoichiometry
* Generating synthetic diffraction or spectroscopy datasets
* Producing training data for machine learning models
* Investigating phase transitions vs temperature or composition
* Testing hypotheses about interaction models
* Rapid prototyping before expensive atomistic simulations

---

## Package Structure
```
nmc_anneal/
    ├── core/       # lattice, energy, annealing
    ├── analysis/   # convergence, phase, NMR
    ├── viz/        # plotting and GUI tools
    ├── io/         # input parsing and configuration
    └── tests/      # automated tests
```

---

## Intended Users
- Battery materials researchers  
- Solid-state chemists  
- NMR spectroscopists  
- Computational materials scientists  

---

## Documentation
More detailed documentation is available in the docs/ directory:
* Theory and physical model
* Algorithm description
* Input file format
* Example workflows
* API reference

___

## Installation

### Install from source (recommended for now)

```bash
git clone https://github.com/yourusername/nmc_anneal
cd nmc_anneal
pip install -e .
```

### Development install
```bash
pip install -e .[dev]
```
___

## Requirements
* Python ≥ 3.10
* NumPy
* Matplotlib (optional, for visualization)
* PyQT (optional, for visualization)

___

## Example workflow

```python
import nmc_anneal as nmc
from pathlib import Path

# Load configuration
config = nmc.parse_input_file(Path("input.txt"))

# Initialize lattice
species, charges = nmc.initialize_lattice(config)

# Run annealing
nmc.anneal_3Dlattice(config, charges, species)
)
```

Then analyze:

* Final configuration
* Energy vs temperature
* Convergence behavior
* Phase structure
* Experimental observables

---


## Performance
Typical simulations of a moderate lattice (e.g., 30×30×4) converge in minutes on a modern laptop. The process takes advantage of the natural translational symmetry for parallel computation in the vertical direction. Best performance will be found when the number of layers is equal to the number of available CPU cores.

## Reproducibility
Simulations are deterministic when a random seed is specified in the configuration file. Energies and average structures are also reproducible in the ergodic limit.


## Citation
The algorithms in this package are a large expansion of the method published in *Structure Solution of Metal-Oxide Li Battery Cathodes from Simulated Annealing and Lithium NMR Spectroscopy* K. J. Harris, J. M. Foster, M. Z. Tessaro, M. Jiang, X. Yang, Y. Wu, B. Protas, and G. R. Goward *Chem. Mater.* 29, 5550−5557, **2017**. Please cite that work if you make use of this code in an academic paper.


## Contributions
Contributions are welcome.
Before submitting a pull request:
* Run formatting: ```black .```
* Run linting: ```ruff check .```
* Run type checking:```mypy src```
* Run tests: ```pytest```

---

## Roadmap
Planned future features include:
* Pair distribution function (PDF) analysis
* Automated fitting to experimental NMR spectra
* Expanded visualization tools

---

## License
MIT License

---
## Project Status
This project is under active development and is currently focused on:
* Core simulation algorithm options
* Scientific validation
* Expanded analysis and visualization tools

---
## Final note

This project exists because **understanding disorder matters**. Real materials are not perfectly ordered or static. If we want to understand and control materials, tools that efficiently explore realistic configurational space are essential for connecting theory, simulation, and experiment.


