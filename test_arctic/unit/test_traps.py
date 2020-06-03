import numpy as np
import pytest
from scipy import integrate
import matplotlib.pyplot as plt
from copy import deepcopy

import arctic as ac


class TestTrapParams:
    def test__parallel_x1__serial_x1_trap___sets_value_correctly(self):

        parallel_trap_0 = ac.Trap(density=0.1, release_timescale=1.0)
        parallel_trap_1 = ac.Trap(density=0.2, release_timescale=2.0)

        serial_trap_0 = ac.Trap(density=0.3, release_timescale=3.0)
        serial_trap_1 = ac.Trap(density=0.4, release_timescale=4.0)

        assert parallel_trap_0.density == 0.1
        assert parallel_trap_0.release_timescale == 1.0
        assert parallel_trap_1.density == 0.2
        assert parallel_trap_1.release_timescale == 2.0

        assert serial_trap_0.density == 0.3
        assert serial_trap_0.release_timescale == 3.0
        assert serial_trap_1.density == 0.4
        assert serial_trap_1.release_timescale == 4.0


#    def test__ccd_class___sets_value_correctly(self):
#
#        parallel_ccd = ac.CCDComplex(
#            full_well_depth=10000.0,
#            well_notch_depth=0.01,
#            well_fill_alpha=0.2,
#            well_fill_power=0.8,
#        )
#        parallel_ccd = ac.CCDPhase(parallel_ccd)
#
#        serial_ccd = ac.CCDComplex(
#            full_well_depth=1000.0,
#            well_notch_depth=1.02,
#            well_fill_alpha=1.1,
#            well_fill_power=1.4,
#        )
#        serial_ccd = ac.CCDPhase(serial_ccd)
#
#        assert parallel_ccd.full_well_depth == 10000.0
#        assert parallel_ccd.well_notch_depth == 0.01
#        assert parallel_ccd.well_fill_alpha == 0.2
#        assert parallel_ccd.well_fill_power == 0.8
#
#        assert serial_ccd.full_well_depth == 1000.0
#        assert serial_ccd.well_notch_depth == 1.02
#        assert serial_ccd.well_range == 998.98
#        assert serial_ccd.well_fill_alpha == 1.1
#        assert serial_ccd.well_fill_power == 1.4


class TestSpecies:
    def test__electrons_released_from_electrons_and_dwell_time(self):

        trap = ac.Trap(release_timescale=1.0)

        assert trap.electrons_released_from_electrons_and_dwell_time(
            electrons=1.0
        ) == pytest.approx(0.6321, 1.0e-4)
        assert trap.electrons_released_from_electrons_and_dwell_time(
            electrons=2.0
        ) == pytest.approx(2.0 * 0.6321, 1.0e-4)

        trap = ac.Trap(release_timescale=2.0)

        assert trap.electrons_released_from_electrons_and_dwell_time(
            electrons=1.0
        ) == pytest.approx(0.39346, 1.0e-4)
        assert trap.electrons_released_from_electrons_and_dwell_time(
            electrons=2.0
        ) == pytest.approx(2.0 * 0.39346, 1.0e-4)

    def test__delta_ellipticity_of_trap(self):

        trap = ac.Trap(density=0.5, release_timescale=2.0)

        assert trap.delta_ellipticity == pytest.approx(0.047378295117617694, 1.0e-5)

    def test__delta_ellipticity_of_arctic_params(self):

        parallel_1_trap = ac.Trap(density=0.1, release_timescale=4.0)
        parallel_2_trap = ac.Trap(density=0.1, release_timescale=4.0)
        serial_1_trap = ac.Trap(density=0.2, release_timescale=2.0)
        serial_2_trap = ac.Trap(density=0.7, release_timescale=7.0)

        trap_manager = ac.TrapManager(traps=[parallel_1_trap], n_pixels=1)
        assert trap_manager.delta_ellipticity == parallel_1_trap.delta_ellipticity

        trap_manager = ac.TrapManager(
            traps=[parallel_1_trap, parallel_2_trap], n_pixels=1
        )
        assert (
            trap_manager.delta_ellipticity
            == parallel_1_trap.delta_ellipticity + parallel_2_trap.delta_ellipticity
        )

        trap_manager = ac.TrapManager(traps=[serial_1_trap], n_pixels=1)
        assert trap_manager.delta_ellipticity == serial_1_trap.delta_ellipticity

        trap_manager = ac.TrapManager(traps=[serial_1_trap, serial_2_trap], n_pixels=1)
        assert (
            trap_manager.delta_ellipticity
            == serial_1_trap.delta_ellipticity + serial_2_trap.delta_ellipticity
        )

        trap_manager = ac.TrapManager(
            traps=[parallel_1_trap, parallel_2_trap, serial_1_trap, serial_2_trap],
            n_pixels=1,
        )

        assert trap_manager.delta_ellipticity == pytest.approx(
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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
                    lambda density: ac.Trap(density=density, release_timescale=1.0),
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


class TestTrapManagerUtilities:
    def test__n_trapped_electrons_from_watermarks(self):
        trap = ac.Trap(density=10, release_timescale=1)
        trap_manager = ac.TrapManager(traps=[trap], n_pixels=6)

        assert (
            trap_manager.n_trapped_electrons_from_watermarks(
                watermarks=trap_manager.watermarks
            )
            == 0
        )

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )

        assert trap_manager.n_trapped_electrons_from_watermarks(
            watermarks=watermarks
        ) == ((0.5 * 0.8 + 0.2 * 0.4 + 0.1 * 0.3) * trap.density)


#        watermarks = np.array(
#            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
#        )
#
#        assert trap_manager.n_trapped_electrons_from_watermarks(
#            watermarks=watermarks, fractional_width=0.5
#        ) == ((0.5 * 0.8 + 0.2 * 0.4 + 0.1 * 0.3) * trap.density * 0.5)


class TestInitialWatermarks:
    def test__initial_watermark_array__uses_n_pixels_and_total_traps_to_set_size(self,):

        # 1 trap species, 3 n_pixels of image pixels
        trap_manager = ac.TrapManager(traps=[ac.Trap()], n_pixels=3)
        assert (trap_manager.watermarks == np.zeros(shape=(7, 2))).all()

        # 5 trap species, 3 n_pixels of image pixels
        trap_manager = ac.TrapManager(
            traps=[ac.Trap(), ac.Trap(), ac.Trap(), ac.Trap(), ac.Trap()], n_pixels=3
        )
        assert (trap_manager.watermarks == np.zeros(shape=(7, 6))).all()

        # 2 trap species, 5 n_pixels of image pixels
        trap_manager = ac.TrapManager(traps=[ac.Trap(), ac.Trap()], n_pixels=5)
        assert (trap_manager.watermarks == np.zeros(shape=(11, 3))).all()


class TestElectronsReleasedAndCapturedInstantCapture:
    def test__empty_release(self):

        traps = [ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        n_electrons_released = trap_manager.n_electrons_released()

        assert n_electrons_released == pytest.approx(0)
        assert np.all(trap_manager.watermarks == 0)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        )

    def test__single_trap_release(self):

        traps = [ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0], [0, 0]]
        )

        n_electrons_released = trap_manager.n_electrons_released()

        assert n_electrons_released == pytest.approx(2.5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__multiple_traps_release(self):

        traps = [
            ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5)),
            ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        trap_manager.watermarks = np.array(
            [
                [0.5, 0.8, 0.3],
                [0.2, 0.4, 0.2],
                [0.1, 0.2, 0.1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        n_electrons_released = trap_manager.n_electrons_released()

        assert n_electrons_released == pytest.approx(2.5 + 1.28)
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [
                    [0.5, 0.4, 0.06],
                    [0.2, 0.2, 0.04],
                    [0.1, 0.1, 0.02],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
               ]
            )
        )

    def test__single_trap_release__change_time(self):

        # Half the time, half the lifetime --> same result
        traps = [
            ac.TrapInstantCapture(density=10, release_timescale=-0.5 / np.log(0.5))
        ]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )

        n_electrons_released = trap_manager.n_electrons_released(dwell_time=0.5)

        assert n_electrons_released == pytest.approx(2.5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]])
        )

    #
    # FRACTIONAL_WIDTH deprecated since it is redundant with full_well_depth
    #
    #    def test__single_trap_release__change_fractional_width(self):
    #
    #        # Half the time, double the density --> same result
    #        traps = [ac.TrapInstantCapture(density=20, release_timescale=-1 / np.log(0.5))] #1.4426950408889634
    #        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)
    #
    #        trap_manager.watermarks = np.array(
    #            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
    #        )
    #
    #        n_electrons_released = trap_manager.n_electrons_released(fractional_width=0.5)
    #
    #        assert n_electrons_released == pytest.approx(2.5)
    #        assert trap_manager.watermarks == pytest.approx(
    #            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]])
    #        )

    def test__first_capture(self):

        n_free_electrons = 2500  # --> cloud_fractional_volume = 0.5

        traps = [ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        ccd = ac.CCDPhase(
            ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7),
            phase=0,
        )

        n_electrons_captured = trap_manager.n_electrons_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        assert n_electrons_captured == pytest.approx(5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__multiple_traps_capture(self):

        n_free_electrons = 3600  # --> cloud_fractional_volume = 0.6

        traps = [
            ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5)),
            ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        trap_manager.watermarks = np.array(
            [
                [0.5, 0.8, 0.7],
                [0.2, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        ccd = ac.CCDPhase(
            ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7),
            phase=0,
        )

        n_electrons_captured = trap_manager.n_electrons_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        assert n_electrons_captured == pytest.approx(
            (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8
        )
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [
                    [0.6, 1, 1],
                    [0.1, 0.4, 0.3],
                    [0.1, 0.3, 0.2],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )
        )

    def test__not_enough__first_capture(self):

        n_free_electrons = (
            2.5e-3  # --> cloud_fractional_volume = 4.9999e-4, enough=enough = 0.50001
        )

        traps = [ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        ccd = ac.CCDPhase(
            ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7),
            phase=0,
        )

        n_electrons_captured = trap_manager.n_electrons_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        assert n_electrons_captured == pytest.approx(0.0025)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[4.9999e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__not_enough__multiple_traps_capture(self):

        n_free_electrons = (
            4.839e-4  # --> cloud_fractional_volume = 2.199545e-4, enough=enough = 0.5
        )

        traps = [
            ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5)),
            ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)

        trap_manager.watermarks = np.array(
            [
                [0.5, 0.8, 0.7],
                [0.2, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        ccd = ac.CCDPhase(
            ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7),
            phase=0,
        )

        n_electrons_captured = trap_manager.n_electrons_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        assert n_electrons_captured == pytest.approx(
            2.199545e-4 * (0.1 * 10 + 0.15 * 8)
        )
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [
                    [2.199545e-4, 0.9, 0.85],
                    [0.5 - 2.199545e-4, 0.8, 0.7],
                    [0.2, 0.4, 0.3],
                    [0.1, 0.3, 0.2],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )
        )

    def test__combined_release_and_capture__multiple_traps(self):

        traps = [
            ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5)),
            ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2)),
        ]
        trap_manager_1 = ac.TrapManagerInstantCapture(traps=traps, n_pixels=6)
        ccd = ac.CCDPhase(
            ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7),
            phase=0,
        )

        n_free_electrons = 3600  # --> cloud_fractional_volume = 0.6

        trap_manager_1.watermarks = np.array(
            [
                [0.5, 0.8, 0.3],
                [0.2, 0.4, 0.2],
                [0.1, 0.2, 0.1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        trap_manager_2 = deepcopy(trap_manager_1)

        # Deprecated separate functions
        n_electrons_released = trap_manager_1.n_electrons_released()
        n_electrons_captured = trap_manager_1.n_electrons_captured(
            n_free_electrons=n_free_electrons + n_electrons_released, ccd=ccd
        )

        # Combined function
        n_electrons_released_and_captured = trap_manager_2.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        # Same net electrons and updated watermarks
        assert (
            n_electrons_released - n_electrons_captured
            == n_electrons_released_and_captured
        )
        assert trap_manager_1.watermarks == pytest.approx(trap_manager_2.watermarks)


class TestTrapManagerTrackTime:
    def test__fill_fraction_from_time_elapsed(self):

        trap = ac.TrapInstantCapture(density=10, release_timescale=2)

        fill = trap.fill_fraction_from_time_elapsed(1)
        assert fill == np.exp(-0.5)

        time_elapsed = trap.time_elapsed_from_fill_fraction(0.5)
        assert time_elapsed == -2 * np.log(0.5)

        assert fill == trap.fill_fraction_from_time_elapsed(
            trap.time_elapsed_from_fill_fraction(fill)
        )
        assert time_elapsed == trap.time_elapsed_from_fill_fraction(
            trap.fill_fraction_from_time_elapsed(time_elapsed)
        )

    def test__watermarks_converted_to_fill_fractions_from_elapsed_times(self):

        trap = ac.TrapInstantCapture(density=10, release_timescale=2)
        trap_manager = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
        watermarks_fill = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        watermarks_time = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )

        assert watermarks_fill == pytest.approx(
            trap_manager.watermarks_converted_to_fill_fractions_from_elapsed_times(
                watermarks_time
            )
        )
        assert watermarks_time == pytest.approx(
            trap_manager.watermarks_converted_to_elapsed_times_from_fill_fractions(
                watermarks_fill
            )
        )
        assert watermarks_fill == pytest.approx(
            trap_manager.watermarks_converted_to_fill_fractions_from_elapsed_times(
                trap_manager.watermarks_converted_to_elapsed_times_from_fill_fractions(
                    watermarks_fill
                )
            )
        )

    def test__n_trapped_electrons_from_watermarks_using_time(self):

        trap = ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))
        trap_manager_fill = ac.TrapManagerInstantCapture(traps=[trap], n_pixels=6)
        trap_manager_time = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)

        trap_manager_fill.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        trap_manager_time.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        n_electrons_fill = trap_manager_fill.n_trapped_electrons_from_watermarks(
            watermarks=trap_manager_fill.watermarks
        )
        n_electrons_time = trap_manager_time.n_trapped_electrons_from_watermarks(
            watermarks=trap_manager_time.watermarks
        )

        assert n_electrons_fill == n_electrons_time

    def test__electrons_released_and_captured_using_time(self):

        n_free_electrons = 5e4  # cloud_fractional_volume ~= 0.656
        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)
        ccd = ac.CCDPhase(ccd)

        trap = ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))
        trap_manager_fill = ac.TrapManagerInstantCapture(traps=[trap], n_pixels=6)
        trap_manager_time = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)

        trap_manager_fill.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        trap_manager_time.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )

        net_electrons_fill = trap_manager_fill.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )
        net_electrons_time = trap_manager_time.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )

        assert net_electrons_fill == net_electrons_time
        assert trap_manager_fill.watermarks == pytest.approx(
            trap_manager_time.watermarks_converted_to_fill_fractions_from_elapsed_times(
                trap_manager_time.watermarks
            )
        )

    def test__electrons_released_and_captured_using_time_multiple_traps(self):
        n_free_electrons = 1e3
        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)
        ccd = ac.CCDPhase(ccd)

        trap_1 = ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))
        trap_2 = ac.TrapInstantCapture(density=10, release_timescale=-2 / np.log(0.5))
        trap_manager_fill = ac.TrapManagerInstantCapture(
            traps=[trap_1, trap_2], n_pixels=6
        )
        trap_manager_time = ac.TrapManagerTrackTime(traps=[trap_1, trap_2], n_pixels=6)

        trap_manager_fill.watermarks = np.array(
            [
                [0.5, 0.8, 0.6,],
                [0.2, 0.4, 0.2,],
                [0.1, 0.3, 0.1,],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        trap_manager_time.watermarks = np.array(
            [
                [
                    0.5,
                    trap_1.time_elapsed_from_fill_fraction(0.8),
                    trap_2.time_elapsed_from_fill_fraction(0.6),
                ],
                [
                    0.2,
                    trap_1.time_elapsed_from_fill_fraction(0.4),
                    trap_2.time_elapsed_from_fill_fraction(0.2),
                ],
                [
                    0.1,
                    trap_1.time_elapsed_from_fill_fraction(0.3),
                    trap_2.time_elapsed_from_fill_fraction(0.1),
                ],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        net_electrons_fill = trap_manager_fill.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )
        net_electrons_time = trap_manager_time.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )

        assert net_electrons_fill == pytest.approx(net_electrons_time)
        assert trap_manager_fill.watermarks == pytest.approx(
            trap_manager_time.watermarks_converted_to_fill_fractions_from_elapsed_times(
                trap_manager_time.watermarks
            )
        )


class TestTrapLifetimeContinuum:
    def test__distribution_of_traps_with_lifetime(self):

        release_timescale_mu = -1 / np.log(0.5)
        release_timescale_sigma = 0.5

        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )

        # Check that the integral from zero to infinity is one
        assert integrate.quad(
            trap.distribution_of_traps_with_lifetime,
            0,
            np.inf,
            args=(trap.release_timescale_mu, trap.release_timescale_sigma),
        )[0] == pytest.approx(1)

    def test__fill_fraction_from_time_elapsed_narrow_continuum(self):

        # Check that narrow continuum gives similar results to single release_timescale
        # Simple trap
        trap = ac.TrapInstantCapture(density=10, release_timescale=1)
        fill_single = trap.fill_fraction_from_time_elapsed(1)

        # Narrow continuum
        release_timescale_mu = 1
        release_timescale_sigma = 0.1

        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
        fill_continuum = trap.fill_fraction_from_time_elapsed(1)

        assert fill_continuum == pytest.approx(fill_single, rel=0.01)

    def test__time_elapsed_from_fill_fraction_narrow_continuum(self):

        # Check that narrow continuum gives similar results to single release_timescale
        # Simple trap
        trap = ac.TrapInstantCapture(density=10, release_timescale=1)
        time_single = trap.time_elapsed_from_fill_fraction(0.5)

        # Narrow continuum
        release_timescale_mu = 1
        release_timescale_sigma = 0.1

        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
        time_continuum = trap.time_elapsed_from_fill_fraction(0.5)

        assert time_continuum == pytest.approx(time_single, rel=0.01)

    def test__fill_fraction_from_time_elapsed_continuum(self):

        release_timescale_mu = 1
        release_timescale_sigma = 0.5

        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )

        fill = trap.fill_fraction_from_time_elapsed(2)
        time_elapsed = trap.time_elapsed_from_fill_fraction(0.4)

        assert fill == pytest.approx(
            trap.fill_fraction_from_time_elapsed(
                trap.time_elapsed_from_fill_fraction(fill)
            )
        )
        assert time_elapsed == pytest.approx(
            trap.time_elapsed_from_fill_fraction(
                trap.fill_fraction_from_time_elapsed(time_elapsed)
            )
        )

    def test__n_trapped_electrons_from_watermarks(self):

        # Single trap
        trap = ac.TrapInstantCapture(density=10, release_timescale=1)
        trap_manager_single = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
        trap_manager_single.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        n_electrons_single = trap_manager_single.n_trapped_electrons_from_watermarks(
            trap_manager_single.watermarks
        )

        # Continua
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        for sigma in [0.1, 1, 2]:
            median = 1
            trap = ac.TrapLifetimeContinuum(
                density=10,
                distribution_of_traps_with_lifetime=trap_distribution,
                release_timescale_mu=median,
                release_timescale_sigma=sigma,
            )
            trap_manager_continuum = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
            trap_manager_continuum.watermarks = np.array(
                [
                    [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                    [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                    [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            )
            n_electrons_continuum = trap_manager_continuum.n_trapped_electrons_from_watermarks(
                trap_manager_continuum.watermarks
            )

            assert n_electrons_continuum == pytest.approx(n_electrons_single)

    def test__electrons_released_and_captured_continuum(self):

        # Check that narrow continuum gives similar results to single traps
        # and that a wider continuum gives somewhat similar results

        n_free_electrons = 5e4  # cloud_fractional_volume ~= 0.656
        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)
        ccd = ac.CCDPhase(ccd)

        # Single trap
        trap = ac.TrapInstantCapture(density=10, release_timescale=1)
        trap_manager_single = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
        trap_manager_single.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        net_electrons_single = trap_manager_single.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )

        # Narrow continuum
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        release_timescale_mu = 1
        release_timescale_sigma = 0.01
        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
        trap_manager_narrow = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
        trap_manager_narrow.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        net_electrons_narrow = trap_manager_narrow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )

        # Continuum
        release_timescale_mu = 1
        release_timescale_sigma = 1
        trap = ac.TrapLifetimeContinuum(
            density=10,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
        trap_manager_continuum = ac.TrapManagerTrackTime(traps=[trap], n_pixels=6)
        trap_manager_continuum.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        net_electrons_continuum = trap_manager_continuum.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd,
        )

        assert net_electrons_narrow == pytest.approx(net_electrons_single, rel=1e-4)
        assert net_electrons_continuum == pytest.approx(net_electrons_single, rel=2)

        assert trap_manager_narrow.watermarks == pytest.approx(
            trap_manager_single.watermarks, rel=1e-4
        )
        assert trap_manager_continuum.watermarks == pytest.approx(
            trap_manager_single.watermarks, rel=0.5
        )

    def test__TrapLogNormalLifetimeContinuum(self):

        release_timescale_mu = -1 / np.log(0.5)
        release_timescale_sigma = 0.5

        trap = ac.TrapLogNormalLifetimeContinuum(
            density=10,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )

        # Check that the integral from zero to infinity is one
        assert integrate.quad(
            trap.distribution_of_traps_with_lifetime,
            0,
            np.inf,
            args=(trap.release_timescale_mu, trap.release_timescale_sigma),
        )[0] == pytest.approx(1)

        # Check the automatic distribution function is set correctly
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        assert trap.distribution_of_traps_with_lifetime(
            1.2345, release_timescale_mu, release_timescale_sigma
        ) == trap_distribution(1.2345, release_timescale_mu, release_timescale_sigma)

    def test__electrons_released_and_captured_compare_continuum_with_distributions_of_single_traps(
        self,
    ):

        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)
        ccd = ac.CCDPhase(ccd)

        density = 10
        release_timescale = 5
        sigma = 1
        linear_min_lifetime = 1e-3
        linear_max_lifetime = 1000
        linear_sample = 10000
        min_log_lifetime = -3
        max_log_lifetime = 5
        log_sample = 1000
        t_elapsed = 1
        dwell_time = 1
        n_free_electrons = 1e4

        # Log-normal distribution
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        # Split into two
        def trap_distribution_a(release_timescale, median, sigma):
            return (
                2
                * np.heaviside(release_timescale - 2 * median, 0)
                * trap_distribution(release_timescale, median, sigma)
            )

        def trap_distribution_b(release_timescale, median, sigma):
            return (
                2
                * np.heaviside(2 * median - release_timescale, 0)
                * trap_distribution(release_timescale, median, sigma)
            )

        # Continuum traps
        trap_continuum = ac.TrapLifetimeContinuum(
            density=density,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale,
            release_timescale_sigma=sigma,
        )
        trap_manager_continuum = ac.TrapManagerTrackTime(
            traps=[trap_continuum], n_pixels=6
        )
        trap_manager_continuum.watermarks = np.array(
            [[0.5, t_elapsed], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],]
        )
        net_electrons_continuum = trap_manager_continuum.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        # Separated continuum traps
        trap_continuum_split_a = ac.TrapLifetimeContinuum(
            density=density / 2,
            distribution_of_traps_with_lifetime=trap_distribution_a,
            release_timescale_mu=release_timescale,
            release_timescale_sigma=sigma,
        )
        trap_continuum_split_b = ac.TrapLifetimeContinuum(
            density=density / 2,
            distribution_of_traps_with_lifetime=trap_distribution_b,
            release_timescale_mu=release_timescale,
            release_timescale_sigma=sigma,
        )
        trap_manager_continuum_split = ac.TrapManagerTrackTime(
            traps=[trap_continuum_split_a, trap_continuum_split_b], n_pixels=6
        )
        trap_manager_continuum_split.watermarks = np.array(
            [[0.5, t_elapsed, t_elapsed], [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3,]
        )
        net_electrons_continuum_split = trap_manager_continuum_split.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        # Equivalent distributions of single traps, linearly spaced
        lifetimes_linear = np.linspace(
            linear_min_lifetime, linear_max_lifetime, linear_sample
        )
        densities_linear = trap_distribution(lifetimes_linear, release_timescale, sigma)
        densities_linear *= density / densities_linear.sum()
        traps_linear = [
            ac.TrapInstantCapture(density=density, release_timescale=release_timescale)
            for density, release_timescale in zip(densities_linear, lifetimes_linear)
        ]
        trap_manager_linear = ac.TrapManagerTrackTime(traps=traps_linear, n_pixels=6)
        trap_manager_linear.watermarks = np.array(
            [
                np.append([0.5], [t_elapsed] * linear_sample),
                [0] * (linear_sample + 1),
                [0] * (linear_sample + 1),
                [0] * (linear_sample + 1),
                [0] * (linear_sample + 1),
                [0] * (linear_sample + 1),
            ]
        )
        net_electrons_linear = trap_manager_linear.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        # Equivalent distributions of single traps, logarithmically spaced
        lifetimes_log = np.logspace(min_log_lifetime, max_log_lifetime, log_sample)
        lifetimes_fractional_widths = np.append(
            np.append(
                np.exp(0.5 * (np.log(lifetimes_log[1]) + np.log(lifetimes_log[0]))),
                np.exp(0.5 * (np.log(lifetimes_log[2:]) + np.log(lifetimes_log[:-2]))),
            ),
            np.exp(0.5 * (np.log(lifetimes_log[-1]) + np.log(lifetimes_log[-2]))),
        )
        densities_log = trap_distribution(lifetimes_log, release_timescale, sigma)
        densities_log *= lifetimes_fractional_widths
        densities_log *= density / densities_log.sum()
        traps_log = [
            ac.TrapInstantCapture(density=density, release_timescale=release_timescale)
            for density, release_timescale in zip(densities_log, lifetimes_log)
        ]
        trap_manager_log = ac.TrapManagerTrackTime(traps=traps_log, n_pixels=6)
        trap_manager_log.watermarks = np.array(
            [
                np.append([0.5], [t_elapsed] * log_sample),
                [0] * (log_sample + 1),
                [0] * (log_sample + 1),
                [0] * (log_sample + 1),
                [0] * (log_sample + 1),
                [0] * (log_sample + 1),
            ]
        )
        net_electrons_log = trap_manager_log.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )
        net_electrons_log_2 = trap_manager_log.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=ccd
        )

        assert net_electrons_continuum == pytest.approx(net_electrons_continuum_split)
        assert net_electrons_continuum == pytest.approx(net_electrons_linear, rel=0.001)
        assert net_electrons_continuum == pytest.approx(net_electrons_log, rel=0.001)

    def test__trails_from_continuum_traps_compare_with_distributions_of_single_traps(
        self,
    ):

        size = 10
        pixels = np.arange(size)
        image_orig = np.zeros((size, 1))
        image_orig[1, 0] = 1e4

        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)

        density = 10
        release_timescale = 5
        sigma = 1
        min_log_lifetime = -3
        max_log_lifetime = 5
        log_sample = 100

        # Log-normal distribution
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        # Continuum traps
        trap_continuum = ac.TrapLifetimeContinuum(
            density=density,
            distribution_of_traps_with_lifetime=trap_distribution,
            release_timescale_mu=release_timescale,
            release_timescale_sigma=sigma,
        )
        image_continuum = ac.add_cti(
            image=image_orig, parallel_traps=[trap_continuum], parallel_ccd=ccd,
        )

        # Equivalent distributions of single traps, logarithmically spaced
        lifetimes_log = np.logspace(min_log_lifetime, max_log_lifetime, log_sample)
        lifetimes_fractional_widths = np.append(
            np.append(
                np.exp(0.5 * (np.log(lifetimes_log[1]) + np.log(lifetimes_log[0]))),
                np.exp(0.5 * (np.log(lifetimes_log[2:]) + np.log(lifetimes_log[:-2]))),
            ),
            np.exp(0.5 * (np.log(lifetimes_log[-1]) + np.log(lifetimes_log[-2]))),
        )
        densities_log = trap_distribution(lifetimes_log, release_timescale, sigma)
        densities_log *= lifetimes_fractional_widths
        densities_log *= density / densities_log.sum()
        traps_log = [
            ac.TrapInstantCapture(density=density, release_timescale=release_timescale)
            for density, release_timescale in zip(densities_log, lifetimes_log)
        ]
        image_log = ac.add_cti(
            image=image_orig, parallel_traps=traps_log, parallel_ccd=ccd,
        )

        assert image_continuum == pytest.approx(image_log)

    def test__plot_trails_from_continuum_traps_different_distributions(self,):

        # Plotting test -- manually set True to make the plot
        do_plot = False
        # do_plot = True

        size = 20
        pixels = np.arange(size)
        image_orig = np.zeros((size, 1))
        image_orig[1, 0] = 1e4

        ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)

        density = 1000
        release_timescale = 3

        # Log-normal distribution
        def trap_distribution(release_timescale, median, sigma):
            return np.exp(
                -((np.log(release_timescale) - np.log(median)) ** 2) / (2 * sigma ** 2)
            ) / (release_timescale * sigma * np.sqrt(2 * np.pi))

        if do_plot:
            plt.figure()

            # Single trap
            trap_single = ac.TrapInstantCapture(
                density=density, release_timescale=release_timescale
            )
            image_single = ac.add_cti(
                image=image_orig, parallel_traps=[trap_single], parallel_ccd=ccd,
            )
            plt.scatter(pixels, image_single[:, 0], c="k", marker=".", label="Single")

            # Pure exponential for comparison
            exp_trail = np.exp(-pixels[2:] / release_timescale)
            exp_trail *= image_single[2, 0] / exp_trail[0]
            plt.plot(pixels[2:], exp_trail, c="k", alpha=0.3)

            # Different sigma scales
            for sigma in [0.1, 0.5, 1, 2]:
                trap_continuum = ac.TrapLifetimeContinuum(
                    density=density,
                    distribution_of_traps_with_lifetime=trap_distribution,
                    release_timescale_mu=release_timescale,
                    release_timescale_sigma=sigma,
                )
                image_continuum = ac.add_cti(
                    image=image_orig, parallel_traps=[trap_continuum], parallel_ccd=ccd,
                )
                plt.plot(
                    pixels, image_continuum[:, 0], label=r"$\sigma = %.1f$" % sigma
                )

            plt.legend()
            plt.yscale("log")
            plt.xlabel("Pixel")
            plt.ylabel("Counts")

            plt.show()


class TestElectronsReleasedAndCapturedIncludingSlowTraps:

    ccd = ac.CCD(well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7)
    ccd = ac.CCDPhase(ccd)

    density = 10
    release_timescale = 1

    # Old-style traps
    traps_instant = [
        ac.TrapInstantCapture(density=density, release_timescale=release_timescale)
    ]
    trap_manager_instant = ac.TrapManagerInstantCapture(traps=traps_instant, n_pixels=6)

    # Fast capture
    traps_fast = [
        ac.Trap(
            density=density, release_timescale=release_timescale, capture_timescale=0
        )
    ]
    trap_manager_fast = ac.TrapManager(traps=traps_fast, n_pixels=3)

    # Slow capture
    traps_slow = [
        ac.Trap(
            density=density, release_timescale=release_timescale, capture_timescale=0.1
        )
    ]
    trap_manager_slow = ac.TrapManager(traps=traps_slow, n_pixels=3)

    def test__collapse_redundant_watermarks(self):

        # None full
        watermarks = self.trap_manager_fast.collapse_redundant_watermarks(
            np.array([[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]])
        )
        assert watermarks == pytest.approx(
            np.array([[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]])
        )

        # None overwritten
        watermarks = self.trap_manager_fast.collapse_redundant_watermarks(
            np.array([[0.5, 1], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]])
        )
        assert watermarks == pytest.approx(
            np.array([[0.5, 1], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]])
        )

        # Some overwritten
        watermarks = self.trap_manager_fast.collapse_redundant_watermarks(
            np.array([[0.5, 1], [0.2, 1], [0.1, 0.2], [0, 0], [0, 0], [0, 0]])
        )
        assert watermarks == pytest.approx(
            np.array([[0.7, 1], [0.1, 0.2], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

        # All overwritten
        watermarks = self.trap_manager_fast.collapse_redundant_watermarks(
            np.array([[0.5, 1], [0.2, 1], [0.1, 1], [0, 0], [0, 0], [0, 0]])
        )
        assert watermarks == pytest.approx(
            np.array([[0.8, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__first_slow_capture(self):

        n_free_electrons = 5e4  # cloud_fractional_volume ~= 0.656

        net_electrons_instant = self.trap_manager_instant.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_fast = self.trap_manager_fast.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_slow = self.trap_manager_slow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )

        # Fast traps reproduce old-style behaviour
        assert self.trap_manager_fast.watermarks == pytest.approx(
            self.trap_manager_instant.watermarks
        )
        assert net_electrons_fast == net_electrons_instant

        # Slow traps capture fewer electrons but same watermark volumes
        assert self.trap_manager_slow.watermarks[:, 0] == pytest.approx(
            self.trap_manager_instant.watermarks[:, 0]
        )
        assert net_electrons_instant < net_electrons_slow

    def test__new_lowest_watermark_slow_capture(self):

        n_free_electrons = 5e3  # cloud_fractional_volume ~= 0.104

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        self.trap_manager_instant.watermarks = deepcopy(watermarks)
        self.trap_manager_fast.watermarks = deepcopy(watermarks)
        self.trap_manager_slow.watermarks = deepcopy(watermarks)

        net_electrons_instant = self.trap_manager_instant.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_fast = self.trap_manager_fast.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_slow = self.trap_manager_slow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )

        # Fast traps reproduce old-style behaviour
        assert self.trap_manager_fast.watermarks == pytest.approx(
            self.trap_manager_instant.watermarks, rel=1e-3
        )
        assert net_electrons_fast == pytest.approx(net_electrons_instant, rel=1e-3)

        # Slow traps capture less than fast
        assert net_electrons_fast < net_electrons_slow

        # Lowest watermark volumes add up to previous volume, fill fractions
        #   increased below the cloud, decreased above it
        assert self.trap_manager_slow.watermarks[:3, 0].sum() == watermarks[0, 0]
        assert (self.trap_manager_slow.watermarks[:1, 1] > watermarks[0, 1]).all()
        assert self.trap_manager_slow.watermarks[2, 1] < watermarks[0, 1]

        # Upper watermark volumes unchanged, fill fractions decreased
        assert self.trap_manager_slow.watermarks[3:, 0] == pytest.approx(
            watermarks[1:-2, 0]
        )
        assert (self.trap_manager_slow.watermarks[3:, 1] <= watermarks[1:-2, 1]).all()

    def test__new_middle_watermark_slow_capture(self):

        n_free_electrons = 5e4  # cloud_fractional_volume ~= 0.656

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        self.trap_manager_instant.watermarks = deepcopy(watermarks)
        self.trap_manager_fast.watermarks = deepcopy(watermarks)
        self.trap_manager_slow.watermarks = deepcopy(watermarks)

        net_electrons_instant = self.trap_manager_instant.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_fast = self.trap_manager_fast.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_slow = self.trap_manager_slow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )

        assert self.trap_manager_fast.watermarks == pytest.approx(
            self.trap_manager_instant.watermarks, rel=1e-3
        )
        assert net_electrons_fast == pytest.approx(net_electrons_instant, rel=1e-3)

        # Slow traps capture less than fast
        assert net_electrons_fast < net_electrons_slow

        # Lowest watermark volume unchanged, fill fractions increased
        assert self.trap_manager_slow.watermarks[0, 0] == watermarks[0, 0]
        assert self.trap_manager_slow.watermarks[0, 1] > watermarks[0, 1]

        # mu watermark volumes add up to previous volume, fill fractions
        #   increased below the cloud, decreased above it
        assert self.trap_manager_slow.watermarks[1:4, 0].sum() == watermarks[1, 0]
        assert (self.trap_manager_slow.watermarks[1:3, 1] > watermarks[1, 1]).all()
        assert self.trap_manager_slow.watermarks[3, 1] < watermarks[1, 1]

        # Upper watermark volumes unchanged, fill fractions decreased
        assert self.trap_manager_slow.watermarks[4:, 0] == pytest.approx(
            watermarks[2:-2, 0]
        )
        assert (self.trap_manager_slow.watermarks[4:, 1] <= watermarks[2:-2, 1]).all()

    def test__new_highest_watermark_slow_capture(self):

        n_free_electrons = 7e4  # cloud_fractional_volume ~= 0.859

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        self.trap_manager_instant.watermarks = deepcopy(watermarks)
        self.trap_manager_fast.watermarks = deepcopy(watermarks)
        self.trap_manager_slow.watermarks = deepcopy(watermarks)

        net_electrons_instant = self.trap_manager_instant.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_fast = self.trap_manager_fast.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_slow = self.trap_manager_slow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )

        # Fast traps reproduce old-style behaviour
        assert self.trap_manager_fast.watermarks == pytest.approx(
            self.trap_manager_instant.watermarks, rel=1e-3
        )
        assert net_electrons_fast == pytest.approx(net_electrons_instant, rel=1e-3)

        # Slow traps capture less than fast
        assert net_electrons_fast < net_electrons_slow

        # Lower watermark volumes unchanged, fill fractions increased
        assert (self.trap_manager_slow.watermarks[:3, 0] == watermarks[:3, 0]).all()
        assert (self.trap_manager_slow.watermarks[:3, 1] > watermarks[:3, 1]).all()

        # New upper watermark volume added, fill fraction increased
        assert self.trap_manager_slow.watermarks[3, 0] > watermarks[3, 0]
        assert self.trap_manager_slow.watermarks[3, 1] > watermarks[3, 1]

    def test__no_available_electrons_slow_capture(self):

        n_free_electrons = 0

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        self.trap_manager_instant.watermarks = deepcopy(watermarks)
        self.trap_manager_fast.watermarks = deepcopy(watermarks)
        self.trap_manager_slow.watermarks = deepcopy(watermarks)

        net_electrons_instant = self.trap_manager_instant.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_fast = self.trap_manager_fast.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )
        net_electrons_slow = self.trap_manager_slow.n_electrons_released_and_captured(
            n_free_electrons=n_free_electrons, ccd=self.ccd, dwell_time=1,
        )

        # Fast traps reproduce old-style behaviour
        assert self.trap_manager_fast.watermarks == pytest.approx(
            self.trap_manager_instant.watermarks, rel=1e-3
        )
        assert net_electrons_fast == pytest.approx(net_electrons_instant, rel=1e-3)

        # Slow traps capture less than fast
        assert net_electrons_fast < net_electrons_slow

        # Lowest watermark volumes add up to previous volume, fill fractions
        #   increased in the new lowest level, decreased above it
        assert self.trap_manager_slow.watermarks[:2, 0].sum() == watermarks[0, 0]
        assert self.trap_manager_slow.watermarks[0, 1] > watermarks[0, 1]
        assert self.trap_manager_slow.watermarks[1, 1] < watermarks[0, 1]

        # Upper watermark volumes unchanged, fill fractions decreased
        assert self.trap_manager_slow.watermarks[2:, 0] == pytest.approx(
            watermarks[1:-1, 0]
        )
        assert (self.trap_manager_slow.watermarks[2:, 1] <= watermarks[1:-1, 1]).all()

        # Fast traps reproduce old-style behaviour
        assert net_electrons_fast == pytest.approx(net_electrons_instant)
        # Slow traps re-capture less so net release slightly more
        assert net_electrons_fast < net_electrons_slow

    def test__updated_watermarks_from_capture_not_enough(self):

        # Initial watermarks with updated volumes to match current watermarks
        watermarks_initial = np.array(
            [[0.5, 0.8], [0.1, 0.4], [0.1, 0.4], [0.1, 0.2], [0, 0], [0, 0]]
        )
        # Initial number of trapped electrons
        trapped_electrons_initial = self.trap_manager_slow.n_trapped_electrons_from_watermarks(
            watermarks=watermarks_initial
        )

        watermarks = np.array(
            [[0.5, 0.9], [0.1, 0.8], [0.1, 0.4], [0.1, 0.2], [0, 0], [0, 0]]
        )
        self.trap_manager_slow.watermarks = watermarks
        # Expected number of trapped electrons
        trapped_electrons_attempted = (
            self.trap_manager_slow.n_trapped_electrons_from_watermarks(
                watermarks=watermarks
            )
            - trapped_electrons_initial
        )

        # But only half the required number of electrons available
        n_free_electrons = 0.5 * trapped_electrons_attempted
        enough = n_free_electrons / trapped_electrons_attempted

        watermarks_not_enough = self.trap_manager_slow.updated_watermarks_from_capture_not_enough(
            self.trap_manager_slow.watermarks, watermarks_initial, enough
        )

        # Filled half-way to their old-style-capture fill fractions
        assert watermarks_not_enough == pytest.approx(
            np.array([[0.5, 0.85], [0.1, 0.6], [0.1, 0.4], [0.1, 0.2], [0, 0], [0, 0]])
        )

        # Resulting number of trapped electrons
        self.trap_manager_slow.watermarks = watermarks_not_enough
        trapped_electrons_final = (
            self.trap_manager_slow.n_trapped_electrons_from_watermarks(
                watermarks=watermarks_not_enough
            )
            - trapped_electrons_initial
        )

        # Only capture the available electrons
        assert trapped_electrons_final == pytest.approx(n_free_electrons)
