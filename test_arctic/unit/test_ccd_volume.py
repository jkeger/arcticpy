import numpy as np
import pytest

import arctic as ac


class TestCCDVolume:
    def test__eletron_fractional_height_from_electrons__gives_correct_value(self,):

        parallel_ccd_volume = ac.CCDVolume(
            well_max_height=10000.0, well_notch_depth=0.0, well_fill_beta=1.0
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=100.0
        )

        assert eletron_fractional_height == 0.01

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=1000.0
        )

        assert eletron_fractional_height == 0.1

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=1000000.0
        )

        assert eletron_fractional_height == 1.0

        parallel_ccd_volume = ac.CCDVolume(
            well_max_height=10000.0, well_notch_depth=0.0, well_fill_beta=0.5
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=100.0
        )

        assert eletron_fractional_height == 0.01 ** 0.5

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=1000.0
        )

        assert eletron_fractional_height == 0.1 ** 0.5

        parallel_ccd_volume = ac.CCDVolume(
            well_max_height=100.0, well_notch_depth=90.0, well_fill_beta=1.0
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=100.0
        )

        assert eletron_fractional_height == 1.0

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=9.0
        )

        assert eletron_fractional_height == 0.0


class TestCCDVolumeComplex:
    def test__eletron_fractional_height_from_electrons__gives_correct_value(self,):

        parallel_ccd_volume = ac.CCDVolumeComplex(
            well_max_height=15.0,
            well_notch_depth=5.0,
            well_fill_alpha=1.0,
            well_fill_beta=1.0,
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=10.0
        )

        assert eletron_fractional_height == 1.0

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=7.5
        )

        assert eletron_fractional_height == 0.5

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=6.0
        )

        assert eletron_fractional_height == 0.2

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=2.0
        )

        assert eletron_fractional_height == 0.0

        parallel_ccd_volume = ac.CCDVolumeComplex(
            well_max_height=15.0,
            well_notch_depth=5.0,
            well_fill_alpha=1.0,
            well_fill_beta=0.5,
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=10.0
        )

        assert eletron_fractional_height == 1.0 ** 0.5

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=7.5
        )

        assert eletron_fractional_height == 0.5 ** 0.5

        parallel_ccd_volume = ac.CCDVolumeComplex(
            well_max_height=15.0,
            well_notch_depth=5.0,
            well_fill_alpha=2.0,
            well_fill_beta=0.5,
        )

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=6.0
        )

        assert eletron_fractional_height == 0.4 ** 0.5

        eletron_fractional_height = parallel_ccd_volume.electron_fractional_height_from_electrons(
            electrons=7.5
        )

        assert eletron_fractional_height == 1.0 ** 0.5
