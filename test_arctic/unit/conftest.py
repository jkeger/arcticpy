import arctic as ac
import numpy as np

import pytest


@pytest.fixture(name="trap_0")
def make_trap_0():
    return ac.Trap(density=10, lifetime=-1 / np.log(0.5))


@pytest.fixture(name="trap_1")
def make_trap_1():
    return ac.Trap(density=8, lifetime=-1 / np.log(0.2))


@pytest.fixture(name="traps_x1")
def make_traps_x1(trap_0):
    return [trap_0]


@pytest.fixture(name="traps_x2")
def make_traps_x2(trap_0, trap_1):
    return [trap_0, trap_1]


@pytest.fixture(name="ccd_volume")
def make_ccd_volume():
    return ac.CCDVolume(
        well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
    )
