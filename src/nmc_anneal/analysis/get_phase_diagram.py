import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import nmc_anneal.core.energy_calculations as encalc
from nmc_anneal.core.anneal_lattice import anneal_3Dlattice
from nmc_anneal.core.config import SimulationConfig

# Type alias for species lattices (dtype="<U4")
SpeciesLattice = NDArray[np.str_]
# Type alias for charges lattices (dtype=np.int8)
ChargesLattice = NDArray[np.int8]


def get_phase_diagram(
    config: SimulationConfig,
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    output_filename: str,
    anneal_type: str,
    n_steps_perT: float,
    sim_start_temp: float,
    sim_end_temp: float,
):
    """
    Generate a phase diagram by annealing at multiple temperatures.

    Performs multiple annealing runs across a temperature range and plots average oxygen
    energy vs. temperature. Useful for understanding thermal behavior and phase transitions.

    Args:
        config (SimulationConfig): Simulation parameters.
        whole_lattice_charges (ChargesLattice): Initial charges lattice.
        whole_lattice_species (SpeciesLattice): Initial species lattice.
        output_filename (str): Path to output PDF file.
        anneal_type (str): "TM Convergence Check" or "Li Convergence Check".
        n_steps_perT (float): Number of annealing steps at each temperature.
        sim_start_temp (float): Starting temperature for the scan.
        sim_end_temp (float): Ending temperature for the scan.

    Raises:
        ValueError: If anneal_type is invalid.
    """

    VALID_ANNEAL_TYPES = {
        "TM Convergence Check",
        "Li Convergence Check",
    }

    if anneal_type not in VALID_ANNEAL_TYPES:
        raise ValueError(
            f"Invalid anneal_type: '{anneal_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ANNEAL_TYPES))}"
        )

    config.curr_conv_check_n_steps = int(n_steps_perT)

    temp_interval = (sim_end_temp - sim_start_temp) / 100
    temp_list = [sim_start_temp + i * temp_interval for i in range(0, 101, 1)]

    starting_lattice_charges = whole_lattice_charges.copy()
    starting_lattice_species = whole_lattice_species.copy()

    energy_trajectories = []
    for run in range(5):
        one_energy_trajectory = []
        for temp in temp_list:
            config.curr_conv_check_hot_temp = temp
            config.curr_conv_check_cold_temp = temp

            whole_lattice_charges = starting_lattice_charges.copy()
            whole_lattice_species = starting_lattice_species.copy()
            anneal_3Dlattice(
                config,
                whole_lattice_charges,
                whole_lattice_species,
                anneal_type,
                graph_energy=False,
            )

            one_energy_trajectory.append(
                encalc.average_all_oxygen_energies(whole_lattice_charges)
            )
        energy_trajectories.append(one_energy_trajectory)
        print(f"Done replicate run {run+1}")

        _plot_temp_trajectories(
            output_filename, energy_trajectories, temp_axis=temp_list
        )


def _plot_temp_trajectories(
    output_filename: str, energy_trajectories: list, temp_axis: list
):
    """
    Plot energy vs. temperature for multiple replicate runs with mean trajectory.

    Creates a scatter plot of individual replicate measurements (black dots) overlaid with
    the mean energy trajectory (red line).

    Args:
        output_filename (str): Path to output PDF or PNG file.
        energy_trajectories (list[np.ndarray]): List of energy arrays, one per replicate run.
        temp_axis (list[float]): Temperature values corresponding to each energy measurement.
    """

    energy_trajectories_arr = np.asarray(energy_trajectories)
    n_checks, n_points = energy_trajectories_arr.shape

    global_ymin = energy_trajectories_arr.min()
    global_ymax = energy_trajectories_arr.max()
    pad = 0.05 * (global_ymax - global_ymin)
    global_ymin -= pad
    global_ymax += pad

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Individual trajectories (black dots)
    for i in range(n_checks):
        ax.plot(
            temp_axis,
            energy_trajectories[i],
            linestyle="None",
            marker="o",
            markersize=9,
            color="black",
            alpha=0.6,  # helps with overplotting
        )

    # Mean trajectory (red line)
    mean_trajectory = energy_trajectories_arr.mean(axis=0)

    ax.plot(
        temp_axis,
        mean_trajectory,
        color="#B22222",
        linewidth=1.0,
        label="Mean trajectory",
        zorder=3,
    )

    ax.set_xlabel("Simulation Temperature")
    ax.set_ylabel(r"$\langle E\rangle_{\mathrm{oxygen}}$")

    ax.set_ylim(global_ymin, global_ymax)

    # Gridlines (5×5 feel without forcing ticks)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.legend()

    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
