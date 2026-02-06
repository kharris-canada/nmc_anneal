from pathlib import Path

import nmc_anneal as nmc
from nmc_anneal.viz.latt_to_img import plot_2Dlattice


def main() -> None:
    # Load stoichiometry and simulation parameters from file
    config = nmc.parse_input_file(Path("examples/ex1_input.txt"))

    # Generate lattice of charges and equivalent lattice of atomic names with randomized positions
    whole_lattice_species, whole_lattice_charges = nmc.initialize_lattice(config)

    # Settle the structure into a very low energy (set with low temp. parameter in input.txt)
    nmc.anneal_3Dlattice(
        config,
        whole_lattice_charges,
        whole_lattice_species,
        anneal_type="Initialize TM",
        graph_energy=False,
    )

    a_2D_TMlayer_to_view = whole_lattice_charges[1, :, :]

    image_filename = "examples/Layer_1.png"
    plot_2Dlattice(a_2D_TMlayer_to_view, image_filename, atom_radius=0.2)


if __name__ == "__main__":
    main()
