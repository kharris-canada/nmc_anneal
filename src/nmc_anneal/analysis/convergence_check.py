import numpy as np
from numpy.typing import NDArray

import nmc_anneal.viz.gridplots as gridplots
from nmc_anneal.core.anneal_lattice import anneal_3Dlattice
from nmc_anneal.core.config import SimulationConfig

# Type alias for species lattices (dtype="<U4")
SpeciesLattice = NDArray[np.str_]
# Type alias for charges lattices (dtype=np.int8)
ChargesLattice = NDArray[np.int8]


def find_and_plot_convergence(
    config: SimulationConfig,
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    output_filename: str,
    anneal_type: str,
    max_n_steps: int,
    sim_hot_temp: float,
    sim_cold_temp: float,
    fraction_max_steps_list: np.ndarray,
):
    """
    Perform annealing at multiple step counts and plot convergence results.

    Anneals the structure for the listed fractions of the total step time (most commonly progressively longer durations)
    and generates a multi-panel plot showing energy trajectories and convergence. Useful for determining sufficient annealing time.

    Args:
        config (SimulationConfig): Simulation parameters (will be modified in-place for curr_conv_check variables).
        whole_lattice_charges (ChargesLattice): Initial charges lattice.
        whole_lattice_species (SpeciesLattice): Initial species lattice.
        output_filename (str): Path to output PDF file.
        anneal_type (str): "TM Convergence Check" or "Li Convergence Check".
        max_n_steps (int): Maximum number of steps to test.
        sim_hot_temp (float): Initial simulated annealing temperature.
        sim_cold_temp (float): Final simulated annealing temperature.
        fraction_max_steps_list (np.ndarray): Fractional multiples of max_n_steps to test (e.g., [0.25, 0.5, 1.0]).

    Raises:
        ValueError: If fraction_max_steps_list has more than 8 elements or if anneal_type is invalid.
    """
    if len(fraction_max_steps_list) > 8:
        raise ValueError(
            "Automatic plotting of convergence only works for up to 9 simulation lengths. "
        )

    VALID_ANNEAL_TYPES = {
        "TM Convergence Check",
        "Li Convergence Check",
    }

    if anneal_type not in VALID_ANNEAL_TYPES:
        raise ValueError(
            f"Invalid anneal_type: '{anneal_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ANNEAL_TYPES))}"
        )

    config.curr_conv_check_max_n_steps = max_n_steps
    config.curr_conv_check_hot_temp = sim_hot_temp
    config.curr_conv_check_cold_temp = sim_cold_temp

    trajectories = []
    step_counts = []
    final_avg_energies = []

    starting_lattice_charges = whole_lattice_charges.copy()
    starting_lattice_species = whole_lattice_species.copy()

    for fraction_max_steps in fraction_max_steps_list:
        config.curr_conv_check_n_steps = round(fraction_max_steps * max_n_steps)

        whole_lattice_charges = starting_lattice_charges.copy()
        whole_lattice_species = starting_lattice_species.copy()

        energy_trajectory = anneal_3Dlattice(
            config,
            whole_lattice_charges,
            whole_lattice_species,
            anneal_type,
            graph_energy=True,
        )

        last_5_percent_start = int(95 * (len(energy_trajectory) / 100))
        avg_final_energy = float(np.mean(energy_trajectory[last_5_percent_start:]))

        trajectories.append(energy_trajectory)
        step_counts.append(config.curr_conv_check_n_steps)
        final_avg_energies.append(avg_final_energy)

        gridplots.plot_energy_convergence_grid(
            config,
            anneal_type,
            trajectories,
            step_counts,
            final_avg_energies,
            output_filename,
        )
