# API Reference

## Core Module

### Configuration

**`parse_input_file(path: Path) → SimulationConfig`**

Parse a key-value input file and return a `SimulationConfig` object. Lines beginning with '#' are ignored.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `Path` | Path to input configuration file |

| Returns | Type | Description |
|---------|------|-------------|
| `config` | `SimulationConfig` | Configuration object with all simulation parameters |

---

### Initialization

**`initialize_lattice(config: SimulationConfig) → tuple[SpeciesLattice, ChargesLattice]`**

Generate initial random NMC structure based on stoichiometry in config.

Even layers (0, 2, 4, ...) are Li layers; odd layers (1, 3, 5, ...) are transition metal (TM) layers. Oxygen layers are implicit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `SimulationConfig` | Simulation parameters with stoichiometry fractions |

| Returns | Type | Description |
|---------|------|-------------|
| `lattice_species` | `SpeciesLattice` | Shape (2×n_layers, width, width), dtype="<U4" |
| `lattice_charges` | `ChargesLattice` | Shape (2×n_layers, width, width), dtype=int8 |

---

### Annealing

**`anneal_3Dlattice(config: SimulationConfig, whole_lattice_charges: ChargesLattice, whole_lattice_species: SpeciesLattice, anneal_type: str, graph_energy: bool = False) → list[float]`**

Run parallel simulated annealing to order the crystal structure. Modifies charge and species arrays in place.

Distributes annealing over n_layers parallel processes. Valid anneal types: "Initialize TM", "Anneal Li", "TM Convergence Check", "Li Convergence Check".

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `SimulationConfig` | Simulation parameters with temperature profiles |
| `whole_lattice_charges` | `ChargesLattice` | 3D charge array (modified in place) |
| `whole_lattice_species` | `SpeciesLattice` | 3D species array (modified in place) |
| `anneal_type` | `str` | Type of annealing to perform |
| `graph_energy` | `bool` | Track energy trajectory if True |

| Returns | Type | Description |
|---------|------|-------------|
| `energies` | `list[float]` | Energy trajectory (100 checkpoints) if `graph_energy=True`, else empty list |

---

### Delithiation

**`delithiate(config: SimulationConfig, whole_lattice_charges: ChargesLattice, whole_lattice_species: SpeciesLattice, frac_li_to_remove: float) → None`**

Remove lithium from the structure and oxidize transition metals for charge balance. Modifies arrays in place.

Uses the oxidation model specified in `config.oxidation_model`. Li finds lowest-energy positions via nearest-neighbor analysis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `SimulationConfig` | Parameters including oxidation model |
| `whole_lattice_charges` | `ChargesLattice` | 3D charge array (modified in place) |
| `whole_lattice_species` | `SpeciesLattice` | 3D species array (modified in place) |
| `frac_li_to_remove` | `float` | Fraction of total Li atoms to remove (0.0–1.0) |

---

## Structure and Property Analysis Module

**`get_all_nmr_shifts(whole_lattice_charges: ChargesLattice, whole_lattice_species: SpeciesLattice, nmr_shifts_dict_90s: dict, nmr_shifts_dict_180s: dict) → tuple[np.ndarray, np.ndarray]`**

Calculate list of <sup>7</sup>Li NMR chemical shifts present in structure along with their multiplicity.

Expects shift dictionaries mapping TM species names to ppm shifts for 90° and 180° bonding geometries.

| Returns | Type | Description |
|---------|------|-------------|
| `shifts` | `np.ndarray` | Unique chemical shift values (ppm) |
| `intensities` | `np.ndarray` | Multiplicity of each shift |

---

**`find_and_plot_convergence(config: SimulationConfig, whole_lattice_charges: ChargesLattice, whole_lattice_species: SpeciesLattice, output_filename: str, anneal_type: str, max_n_steps: int, sim_hot_temp: float, sim_cold_temp: float, fraction_max_steps_list: np.ndarray) → None`**

Generate convergence diagnostics by running annealing with different step counts.

Produces a 3×3 PDF grid showing energy trajectories and convergence behavior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `anneal_type` | `str` | "TM Convergence Check" or "Li Convergence Check" |
| `max_n_steps` | `int` | Maximum number of MC steps |
| `sim_hot_temp` | `float` | Starting temperature (K) |
| `sim_cold_temp` | `float` | Ending temperature (K) |
| `fraction_max_steps_list` | `np.ndarray` | Fractions of max steps to sample (max 8 values) |
| `output_filename` | `str` | Output PDF file path |

---

**`get_phase_diagram(config: SimulationConfig, whole_lattice_charges: ChargesLattice, whole_lattice_species: SpeciesLattice, output_filename: str, anneal_type: str, n_steps_perT: float, sim_start_temp: float, sim_end_temp: float) → None`**

Generate energy vs. temperature plot by running annealing over a temperature range.

Useful for identifying ordering transitions and determining optimal simulation temperatures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_steps_perT` | `float` | MC steps between temperature checkpoints |
| `sim_start_temp` | `float` | Temperature to start sweep (K) |
| `sim_end_temp` | `float` | Temperature to end sweep (K) |
| `output_filename` | `str` | Output PDF file path |

---

## Visualization Module (Optional)

**`generate_spectrum(shifts: np.ndarray, intensities: np.ndarray, percent_gaussian: float, fwhm_at_zero: float, fwhm_linear_scale: float, n_points: int, xmin: float, xmax: float) → tuple[np.ndarray, np.ndarray]`**

Generate synthetic <sup>7</sup>Li MAS-NMR spectrum from peak shifts and intensities.

Uses pseudo-Voigt line shape with variable Gaussian/Lorentzian mixing and field-dependent broadening.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shifts` | `np.ndarray` | NMR shifts (ppm) for each site |
| `intensities` | `np.ndarray` | Multiplicity of each peak |
| `percent_gaussian` | `float` | Gaussian fraction of lineshape (0–100%) |
| `fwhm_at_zero` | `float` | Linewidth at 0 ppm (Hz) |
| `fwhm_linear_scale` | `float` | Additional broadening per ppm offset (Hz/ppm) |
| `n_points` | `int` | Number of digital points |
| `xmin`, `xmax` | `float` | Spectral range (ppm) |

| Returns | Type | Description |
|---------|------|-------------|
| `x` | `np.ndarray` | Chemical shift axis (ppm) |
| `y` | `np.ndarray` | Intensity values |

---

**`run_peak_gui(datasets: dict) → None`**

Launch interactive GUI for adjusting NMR lineshape parameters and comparing simulated spectrum to experiment.

Provides sliders to adjust Gaussian/Lorentzian mixing, linewidth at zero shift, field-dependent broadening, and spectral range. Useful for fitting simulated spectra to experimental data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasets` | `dict` | Dictionary mapping dataset names (e.g., "Simulation", "Experiment") to tuples of (shifts, intensities). Shifts are ppm values, intensities are multiplicities. |

**Requirements:** PyQt5, matplotlib

---

## Data Types

**`SpeciesLattice = NDArray[np.str_]`**

3D array of atomic species names, dtype="<U4" (4-character Unicode strings). Shape is (2×n_layers, width, width).

**`ChargesLattice = NDArray[np.int8]`**

3D array of atomic charges, dtype=int8. Shape is (2×n_layers, width, width).

**`SimulationConfig`**

Data class containing all simulation parameters (from input file or set programmatically). See `core.config` for full parameter list.
