from pathlib import Path
import pytest
import numpy as np

import nmc_anneal as nmc
from nmc_anneal import SimulationConfig


def test_populate_lattice_species_and_charges():
    config = SimulationConfig(
        width=6,
        n_layers=2,
        li_fraction_li_layer=1.0,
        mn_fraction_li_layer=0.0,
        ni2_fraction_li_layer=0.0,
        ni3_fraction_li_layer=0.0,
        vac_fraction_li_layer=0.0,
        li_fraction_tm_layer=0.0,
        mn_fraction_tm_layer=0.4,
        ni2_fraction_tm_layer=0.3,
        ni3_fraction_tm_layer=0.1,
        co_fraction_tm_layer=0.2,
        vac_fraction_tm_layer=0.0,
        initialize_anneal_steps=1e4,
        initialize_anneal_hot_temp=10,
        initialize_anneal_cold_temp=1,
        delithiation_steps=1,
        delithiation_fraction_to_remove=0.1,
        oxidation_model="ni_2to4",
        mid_delithiation_anneal_steps=1.5e3,
        mid_delithiation_anneal_hot_temp=0.0,
        mid_delithiation_anneal_cold_temp=0.0,
        output_file=None,
        random_seed=0,
    )

    species, charges = nmc.initialize_lattice(config)

    assert species.shape == charges.shape
    assert species.dtype.kind == "U"
    assert charges.dtype == np.int8
    assert np.all(charges[species == "Ni"] == 2)
