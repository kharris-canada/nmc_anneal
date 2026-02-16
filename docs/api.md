# API Reference 

## Configuration

### parse_input_file(path)
Reads simulation configuration file and returns config object.

---

## Initialization

### initialize_lattice(config)
Generates initial lattice configuration.

Returns:
- species array
- charge array

---

## Simulation

### anneal_3Dlattice(config, charges, species)
Runs simulated annealing to produce ordered/disordered structure.

---

### run_delithiation(config, species, charges)
Simulates lithium extraction.

---

## Analysis

### generate_nmr_spectrum(species, charges)
Returns synthetic lithium NMR spectrum.

---

### find_and_plot_convergence(path)
Generates convergence diagnostics from simulation output.

---

## Visualization

### plot_spectrum(spectrum)
Plots NMR spectrum.

---

More detailed docstrings are available in the source code.
