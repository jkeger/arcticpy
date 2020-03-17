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


class TestMultiPhase:
    def test__mutli_phase_initialisation(self):
        # All duplicated
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01,
            well_fill_beta=0.8,
            well_max_height=84700,
            phase_widths=[0.5, 0.2, 0.2, 0.1],
        )

        assert ccd_volume.well_notch_depth == [0.01] * 4
        assert ccd_volume.well_fill_beta == [0.8] * 4
        assert ccd_volume.well_max_height == [84700] * 4
        assert ccd_volume.well_range == [84700 - 0.01] * 4

        # Some duplicated
        ccd_volume = ac.CCDVolume(
            well_notch_depth=[0.01],
            well_fill_beta=[0.8],
            well_max_height=[84700, 1e5, 2e5, 3e5],
            phase_widths=[0.5, 0.2, 0.2, 0.1],
        )

        assert ccd_volume.well_notch_depth == [0.01] * 4
        assert ccd_volume.well_fill_beta == [0.8] * 4
        assert ccd_volume.well_max_height == [84700, 1e5, 2e5, 3e5]
        assert ccd_volume.well_range == [
            84700 - 0.01,
            1e5 - 0.01,
            2e5 - 0.01,
            3e5 - 0.01,
        ]

    def test__extract_phase(self):
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01,
            well_fill_beta=0.8,
            well_max_height=[84700, 1e5, 2e5, 3e5],
            phase_widths=[0.5, 0.2, 0.2, 0.1],
        )

        ccd_phase_0 = ccd_volume.extract_phase(0)
        ccd_phase_1 = ccd_volume.extract_phase(1)
        ccd_phase_2 = ccd_volume.extract_phase(2)
        ccd_phase_3 = ccd_volume.extract_phase(3)

        assert ccd_phase_0.well_notch_depth == 0.01
        assert ccd_phase_0.well_max_height == 84700
        assert ccd_phase_1.well_notch_depth == 0.01
        assert ccd_phase_1.well_max_height == 1e5
        assert ccd_phase_2.well_notch_depth == 0.01
        assert ccd_phase_2.well_max_height == 2e5
        assert ccd_phase_3.well_notch_depth == 0.01
        assert ccd_phase_3.well_max_height == 3e5
