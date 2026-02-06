from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nmc_anneal import SimulationConfig
from nmc_anneal import anneal_3Dlattice
import nmc_anneal.core.energy_calculations as encalc


def get_phase_diagram(
    config: SimulationConfig,
    whole_lattice_charges: np.ndarray,
    whole_lattice_species: np.ndarray,
    output_filename: str,
    anneal_type: str,
    n_steps_perT: float,
    sim_start_temp: float,
    sim_end_temp: float,
):
    """
    Reads in: parameters and existing lattice (expected to match SimulationConfig variables)
    Anneals multiple runs along the temperature trajectory and generates a plot of the energies vs temperature
    Could be run in cold-to-hot or hot-to-cold direction

    NOTE: you can view the output pdf as it runs

    :param config: Dict style data class containing most simulation parameters. See parser.py
    :type config: SimulationConfig
    :param whole_lattice_charges: Array containing all charges in structure in the correct geometry. Formatted as in initialize_lattice.py.
    :type whole_lattice_charges: np.ndarray
    :param whole_lattice_species: Description
    :type whole_lattice_species: Array containing names of all ions in structure in the correct geometry. Formatted as in initialize_lattice.py.
    :param output_filename: Name of file to write graphic to (name.pdf|name.png)
    :type output_filename: str
    :param anneal_type: Which layer types to anneal (TM Convergence Check|Li Convergence Check)
    :type anneal_type: str
    :param n_steps_perT: Number of simulated annealing steps between temperatures
    :type n_steps_perT: float
    :param sim_start_temp: Simulated annealing start T
    :type sim_start_temp: float
    :param sim_end_temp: Simulated annealing ending T
    :type sim_end_temp: float
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


import numpy as np
import matplotlib.pyplot as plt


def _plot_temp_trajectories(
    output_filename: str, energy_trajectories: list, temp_axis: list
):
    """
    Make a dot-style plot  of energy vs temperature with several overlapping replicate runs
    plus a red line showing the average trajectory.

    :param output_filename: Name of file to write graphic to (name.pdf|name.png)
    :type output_filename: str
    :param energy_trajectories: list containing the energies measured at each temperatures
    :type energy_trajectories: list
    :param temp_axis: List of the actual temperatures simulations run on
    :type temp_axis: list
    """

    energy_trajectories = np.asarray(energy_trajectories)
    n_checks, n_points = energy_trajectories.shape

    global_ymin = energy_trajectories.min()
    global_ymax = energy_trajectories.max()
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
    mean_trajectory = energy_trajectories.mean(axis=0)

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
