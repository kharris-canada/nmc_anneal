from pathlib import Path

import numpy as np
import pytest

import nmc_anneal as nmc
import nmc_anneal.core.charging_methods as cm


# Test for conservation of charge during delithiation with ni_2to4 method
def test_charge_conv_delith(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 20
    n_layers = 8
    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e4
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    oxidation_model = ni_2to4_co_3to4

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

    nmc.delithiate(config, whole_lattice_charges, whole_lattice_species, 0.1)

    total_charge = whole_lattice_charges.sum()
    tot_num_metals = config.width * config.width * config.n_layers * 2
    avg_charge = total_charge / tot_num_metals

    assert avg_charge == pytest.approx(2, rel=1e-6)


# Test delithiation stoichiometry with ni_2to4_co_3to4 method:
def test_delith_ni2_first(tmp_path: Path) -> None:
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
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    oxidation_model = ni_2to4_co_3to4

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

    cm.delithiate(config, whole_lattice_charges, whole_lattice_species, 0.75)

    assert np.count_nonzero(whole_lattice_species == "Co3+") == 144
    assert np.count_nonzero(whole_lattice_species == "Co4+") == 48
    assert np.count_nonzero(whole_lattice_species == "Ni4+") == 192
    assert np.count_nonzero(whole_lattice_species == "Li") == 144


# Test delithiation stoichiometry with ni_2to4_any_3to4 method:
def test_delith_ni3withco3(tmp_path: Path) -> None:
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
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    oxidation_model = ni_2to3_any_3to4

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

    cm.delithiate(config, whole_lattice_charges, whole_lattice_species, 0.65)

    total_plus4 = np.count_nonzero(whole_lattice_species == "Co4+") + np.count_nonzero(
        whole_lattice_species == "Ni4+"
    )

    assert total_plus4 == 184
    assert np.count_nonzero(whole_lattice_species == "Ni2+") < 2


# Test delithiation stoichiometry with ni_2to3_ni_3to4_co_3to4 method:
def test_delith_twostepNi(tmp_path: Path) -> None:
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
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    oxidation_model = ni_2to3_ni_3to4_co_3to4

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

    cm.delithiate(config, whole_lattice_charges, whole_lattice_species, 0.71)

    assert np.count_nonzero(whole_lattice_species == "Co3+") == 166
    assert np.count_nonzero(whole_lattice_species == "Co4+") == 26
    assert np.count_nonzero(whole_lattice_species == "Ni4+") == 192
    assert np.count_nonzero(whole_lattice_species == "Li") == 166
    assert np.count_nonzero(whole_lattice_species == "Ni2+") < 2
    assert np.count_nonzero(whole_lattice_species == "Ni3+") < 2
