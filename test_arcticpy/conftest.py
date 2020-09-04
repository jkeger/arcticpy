from os import path
import pytest
from autoconf import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=path.join(directory, "config"),
    )
