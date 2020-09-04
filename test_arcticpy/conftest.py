from os import path

import numpy as np
import pytest

import autoarray as aa
from autoconf import conf
from autoarray.fit import fit

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=path.join(directory, "config"),
    )
