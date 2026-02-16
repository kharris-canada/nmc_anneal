from pathlib import Path

import nmc_anneal as nmc
from nmc_anneal.analysis.convergence_check import find_and_plot_convergence


def main() -> None:
    # Load stoichiometry and simulation parameters from file
    config = nmc.parse_input_file(Path("examples/ex1_parameters.txt"))

    # Generate lattice of charges and equivalent lattice of atomic names with randomized positions
    whole_lattice_species, whole_lattice_charges = nmc.initialize_lattice(config)

    # Test out a series of annealing lengths up to length max_n_steps at temperatures that over-ride the input text file
    find_and_plot_convergence(
        config,
        whole_lattice_charges,
        whole_lattice_species,
        output_filename="examples/ex1_energy_convergence.png",
        anneal_type="TM Convergence Check",
        max_n_steps=1e9,
        sim_hot_temp=1.0,
        sim_cold_temp=0.0,
        fraction_max_steps_list=[0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    )


if __name__ == "__main__":
    main()
