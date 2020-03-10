import numpy as np
import pytest

import arctic as ac


class TestSpecies:
    def test__electrons_released_from_electrons_and_time(self):

        trap = ac.Trap(lifetime=1.0)

        assert trap.electrons_released_from_electrons_and_time(
            electrons=1.0
        ) == pytest.approx(0.6321, 1.0e-4)
        assert trap.electrons_released_from_electrons_and_time(
            electrons=2.0
        ) == pytest.approx(2.0 * 0.6321, 1.0e-4)

        trap = ac.Trap(lifetime=2.0)

        assert trap.electrons_released_from_electrons_and_time(
            electrons=1.0
        ) == pytest.approx(0.39346, 1.0e-4)
        assert trap.electrons_released_from_electrons_and_time(
            electrons=2.0
        ) == pytest.approx(2.0 * 0.39346, 1.0e-4)

    def test__delta_ellipticity_of_trap(self):

        trap = ac.Trap(density=0.5, lifetime=2.0)

        assert trap.delta_ellipticity == pytest.approx(
            0.047378295117617694, 1.0e-5
        )

    def test__delta_ellipticity_of_arctic_params(self):

        parallel_1_trap = ac.Trap(density=0.1, lifetime=4.0)
        parallel_2_trap = ac.Trap(density=0.1, lifetime=4.0)
        serial_1_trap = ac.Trap(density=0.2, lifetime=2.0)
        serial_2_trap = ac.Trap(density=0.7, lifetime=7.0)

        parameters = ac.ArcticParams(parallel_traps=[parallel_1_trap])
        assert parameters.delta_ellipticity == parallel_1_trap.delta_ellipticity

        parameters = ac.ArcticParams(
            parallel_traps=[parallel_1_trap, parallel_2_trap]
        )
        assert (
            parameters.delta_ellipticity
            == parallel_1_trap.delta_ellipticity
            + parallel_2_trap.delta_ellipticity
        )

        parameters = ac.ArcticParams(serial_traps=[serial_1_trap])
        assert parameters.delta_ellipticity == serial_1_trap.delta_ellipticity

        parameters = ac.ArcticParams(
            serial_traps=[serial_1_trap, serial_2_trap]
        )
        assert (
            parameters.delta_ellipticity
            == serial_1_trap.delta_ellipticity + serial_2_trap.delta_ellipticity
        )

        parameters = ac.ArcticParams(
            parallel_traps=[parallel_1_trap, parallel_2_trap],
            serial_traps=[serial_1_trap, serial_2_trap],
        )

        assert parameters.delta_ellipticity == pytest.approx(
            parallel_1_trap.delta_ellipticity
            + parallel_2_trap.delta_ellipticity
            + serial_1_trap.delta_ellipticity
            + serial_2_trap.delta_ellipticity,
            1.0e-6,
        )


class TestParallelDensityVary:
    def test_1_trap__density_01__1000_column_pixels__1_row_pixel_so_100_traps__poisson_density_near_01(
        self,
    ):  #

        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (0.1,),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [0.098]

    def test__1_trap__density_1__1000_column_pixels_so_1000_traps__1_row_pixel__poisson_value_is_near_1(
        self,
    ):
        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (1.0,),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [0.992]

    def test__1_trap__density_1___2_row_pixels__poisson_value_is_near_1(self):
        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (1.0,),
                )
            ),
            shape=(1000, 2),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [0.992, 0.962]

    def test__2_trap__1_row_pixel__poisson_for_each_trap_drawn(self):
        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (1.0, 2.0),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [0.992, 1.946]

    def test__2_trap__2_row_pixel__poisson_for_each_trap_drawn(self):
        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (1.0, 2.0),
                )
            ),
            shape=(1000, 2),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [
            0.992,
            1.946,
            0.968,
            1.987,
        ]

    def test__same_as_above_but_3_trap_and_new_values(self):
        parallel_vary = ac.Trap.poisson_trap(
            trap=list(
                map(
                    lambda density: ac.Trap(density=density, lifetime=1.0),
                    (1.0, 2.0, 0.1),
                )
            ),
            shape=(1000, 3),
            seed=1,
        )

        assert [trap.density for trap in parallel_vary] == [
            0.992,
            1.946,
            0.09,
            0.991,
            1.99,
            0.098,
            0.961,
            1.975,
            0.113,
        ]


class TestCCDVolume:
    def test__eletron_fractional_height_from_electrons__gives_correct_value(
        self,
    ):

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
    def test__eletron_fractional_height_from_electrons__gives_correct_value(
        self,
    ):

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


class TestArcticParams:
    def test__parallel_x1__serial_x1_trap___sets_value_correctly(self):

        parallel_trap_0 = ac.Trap(density=0.1, lifetime=1.0)
        parallel_trap_1 = ac.Trap(density=0.2, lifetime=2.0)

        serial_trap_0 = ac.Trap(density=0.3, lifetime=3.0)
        serial_trap_1 = ac.Trap(density=0.4, lifetime=4.0)

        parameters = ac.ArcticParams(
            parallel_traps=[parallel_trap_0, parallel_trap_1],
            serial_traps=[serial_trap_0, serial_trap_1],
        )

        assert type(parameters) == ac.ArcticParams
        assert type(parameters.parallel_traps[0]) == ac.Trap
        assert type(parameters.parallel_traps[1]) == ac.Trap

        assert parameters.parallel_traps[0].density == 0.1
        assert parameters.parallel_traps[0].lifetime == 1.0
        assert parameters.parallel_traps[1].density == 0.2
        assert parameters.parallel_traps[1].lifetime == 2.0

        assert type(parameters) == ac.ArcticParams
        assert type(parameters.serial_traps[0]) == ac.Trap
        assert type(parameters.serial_traps[1]) == ac.Trap

        assert parameters.serial_traps[0].density == 0.3
        assert parameters.serial_traps[0].lifetime == 3.0
        assert parameters.serial_traps[1].density == 0.4
        assert parameters.serial_traps[1].lifetime == 4.0

    def test__ccd_volume_class___sets_value_correctly(self):

        parallel_ccd_volume = ac.CCDVolumeComplex(
            well_max_height=10000.0,
            well_notch_depth=0.01,
            well_fill_alpha=0.2,
            well_fill_beta=0.8,
        )

        serial_ccd_volume = ac.CCDVolumeComplex(
            well_max_height=1000.0,
            well_notch_depth=1.02,
            well_fill_alpha=1.1,
            well_fill_beta=1.4,
        )

        parameters = ac.ArcticParams(
            parallel_ccd_volume=parallel_ccd_volume,
            serial_ccd_volume=serial_ccd_volume,
        )

        assert type(parameters) == ac.ArcticParams
        assert type(parameters.parallel_ccd_volume) == ac.CCDVolumeComplex

        assert parameters.parallel_ccd_volume.well_max_height == 10000.0
        assert parameters.parallel_ccd_volume.well_notch_depth == 0.01
        assert parameters.parallel_ccd_volume.well_range == 9999.99
        assert parameters.parallel_ccd_volume.well_fill_alpha == 0.2
        assert parameters.parallel_ccd_volume.well_fill_beta == 0.8

        assert type(parameters) == ac.ArcticParams
        assert type(parameters.serial_ccd_volume) == ac.CCDVolumeComplex

        assert parameters.serial_ccd_volume.well_max_height == 1000.0
        assert parameters.serial_ccd_volume.well_notch_depth == 1.02
        assert parameters.serial_ccd_volume.well_range == 998.98
        assert parameters.serial_ccd_volume.well_fill_alpha == 1.1
        assert parameters.serial_ccd_volume.well_fill_beta == 1.4
