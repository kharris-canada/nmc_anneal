from pathlib import Path
import pytest

import nmc_anneal as nmc
from nmc_anneal.core.energy_calculations import average_all_oxygen_energies


# Test energy criterion is 0 for ideal Li2MnO3 configuration:
def test_ideal_struct(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 12
    n_layers = 4

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0.3333
    mn_fraction_tm_layer = 0.6666
    ni2_fraction_tm_layer = 0.0
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.0
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    config = nmc.parse_input_file(input_file)

    whole_lattice_species, whole_lattice_charges = nmc.initialize_lattice(config)

    # Perfect 12x12 honeycomb lattice for Li2MnO3 (Li(1/3), Mn(2/3) in TM layer)
    whole_lattice_charges[1, :, :] = [
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
    ]

    whole_lattice_charges[3, :, :] = whole_lattice_charges[1, :, :]
    whole_lattice_charges[5, :, :] = whole_lattice_charges[1, :, :]
    whole_lattice_charges[7, :, :] = whole_lattice_charges[1, :, :]

    energy = average_all_oxygen_energies(whole_lattice_charges)

    assert energy == pytest.approx(0.0)


# Test energy criterion on a specific set of defects
def test_specific_random_struct(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 12
    n_layers = 4
    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0.3333
    mn_fraction_tm_layer = 0.6666
    ni2_fraction_tm_layer = 0.0
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.0
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    config = nmc.parse_input_file(input_file)

    whole_lattice_species, whole_lattice_charges = nmc.initialize_lattice(config)

    whole_lattice_charges[1, :, :] = [
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 40],
        [4, 12, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 6, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 8, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 9, 1, 4, 4],
        [4, 14, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 6, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 40, 11, 4, 4, 6, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
    ]

    whole_lattice_charges[5, :, :] = [
        [4, 1, 4, 4, 1, 4, 4, 1, 16, 4, 1, 4],
        [4, 7, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 6, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 11, 4, 1, 4, 7, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 2, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1],
        [1, 6, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
        [4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4],
        [4, 22, 17, 4, 4, 3, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4],
    ]

    whole_lattice_charges[3, :, :] = whole_lattice_charges[1, :, :]
    whole_lattice_charges[7, :, :] = whole_lattice_charges[1, :, :]

    energy = average_all_oxygen_energies(whole_lattice_charges)

    assert energy == pytest.approx(0.3836805, rel=1e-6)
