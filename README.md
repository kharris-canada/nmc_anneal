# nmc_anneal

**nmc_anneal** is a Python package for generating and analyzing atomistic models of layered nickel–manganese–cobalt (NMC) battery cathodes using charge-balance-driven simulated annealing.

The software predicts lithium ordering, transition-metal disorder, electrochemical delithiation behavior, and synthetic **⁷Li MAS NMR spectra** from atomistic lattice models, enabling direct comparison between structure and experiment.

---

## What This Software Does

- Generates 3D atomistic models of layered NMC cathodes  
- Simulates disorder formation during synthesis using simulated annealing  
- Predicts lithium extraction (delithiation) behavior  
- Generates synthetic **⁷Li MAS NMR spectra** from atomistic structures  
- Enables structure ↔ spectroscopy interpretation  
- Produces convergence diagnostics and phase information  

---

## Intended Users

- Battery materials researchers  
- Solid-state chemists  
- NMR spectroscopists  
- Computational materials scientists  

---

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

## Quick Example
```python
import nmc_anneal as nmc
from pathlib import Path

# Load configuration
config = nmc.parse_input_file(Path("input.txt"))

# Initialize lattice
species, charges = nmc.initialize_lattice(config)

# Run annealing
nmc.anneal_3Dlattice(config, charges, species)
```

## Example Output
The software produces:
* Convergence diagnostics
* Phase diagrams
* Structure images
* Synthetic NMR spectra
* Structure files for further analysis
Example figures are available in the examples/ directory.

## Model Overview
* Structures are represented as a 3D NumPy lattice
* Energy is defined using local oxygen charge balance
* Simulated annealing models disorder formation during synthesis
* A deterministic algorithm models electrochemical delithiation
* Local lithium environments are used to generate synthetic ⁷Li MAS NMR spectra

## Package Structure
```
nmc_anneal/
    ├── core/       # lattice, energy, annealing
    ├── analysis/   # convergence, phase, NMR
    ├── viz/        # plotting and GUI tools
    ├── io/         # input parsing and configuration
    └── tests/      # automated tests
```


## Performance
Typical simulations of a moderate lattice (e.g., 30×30×4) converge in minutes on a modern laptop. The process takes advantage of the natural translational symmetry for parallel computation in the vertical direction. Best performance will be found when the number of layers is equal to the number of available CPU cores.

## Reproducibility
Simulations are deterministic when a random seed is specified in the configuration file. Energies and average structures are also reproducible in the ergodic limit.

## Documentation
More detailed documentation is available in the docs/ directory:
* Theory and physical model
* Algorithm description
* Input file format
* Example workflows
* API reference

## Citation
The algorithms in this package are a large expansion of the method published in *Structure Solution of Metal-Oxide Li Battery Cathodes from Simulated Annealing and Lithium NMR Spectroscopy* K. J. Harris, J. M. Foster, M. Z. Tessaro, M. Jiang, X. Yang, Y. Wu, B. Protas, and G. R. Goward *Chem. Mater.* 29, 5550−5557, **2017**. Please cite that work if you make use of this code in an academic paper.

## Requirements
* Python ≥ 3.10
* NumPy
* Matplotlib (optional, for visualization)
* PyQT (optional, for visualization)

## Contributions
Contributions are welcome.
Before submitting a pull request:
* Run formatting: ```black .```
* Run linting: ```ruff check .```
* Run type checking:```mypy src```
* Run tests: ```pytest```

## Roadmap
Planned future features include:
* Pair distribution function (PDF) analysis
* Automated fitting to experimental NMR spectra
* Expanded visualization tools

## License
MIT License


## Project Status
This project is under active development and is currently focused on:
* Core simulation algorithm options
* Scientific validation
* Expanded analysis and visualization tools