from pathlib import Path

import nmc_anneal as nmc
from nmc_anneal.analysis.struct2nmr import get_all_nmr_shifts
from nmc_anneal.viz.nmr_simpleplot import image_from_peaklist


def main() -> None:
    # Load stoichiometry and simulation parameters from file
    config = nmc.parse_input_file(Path("examples/ex2_parameters.txt"))

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

    # Generate a list of 7Li NMR peak positions and their intensity (site multiplicity)
    nmr_shifts_dict_90s = {
        "Mn": 255,
        "Ni2+": -25,
    }

    nmr_shifts_dict_180s = {
        "Mn": -52,
        "Ni2+": 120,
    }

    nmr_shifts_dict_inlayer = {
        "Mn": 255,
        "Ni2+": -25,
    }

    nmr_ppm_shifts = get_all_nmr_shifts(
        whole_lattice_charges,
        whole_lattice_species,
        nmr_shifts_dict_90s,
        nmr_shifts_dict_180s,
        nmr_shifts_dict_inlayer,
    )

    image_from_peaklist(
        data=nmr_ppm_shifts,
        percent_gaussian=5,
        fwhm_at_zero=3,
        fwhm_linear_scale=0.0,
        output_filename="examples/nmr_simulation.png",
        n_points=4192,
        xmin=-300,
        xmax=1000,
    )


if __name__ == "__main__":
    main()
