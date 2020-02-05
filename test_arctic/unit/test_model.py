import numpy as np
import pytest

import arctic as ac


class TestSpecies:
    def test__electrons_released_from_electrons(self):

        trap = ac.Trap(lifetime=1.0)

        assert trap.electrons_released_from_electrons(
            electrons=1.0
        ) == pytest.approx(0.6321, 1.0e-4)
        assert trap.electrons_released_from_electrons(
            electrons=2.0
        ) == pytest.approx(2.0 * 0.6321, 1.0e-4)

        trap = ac.Trap(lifetime=2.0)

        assert trap.electrons_released_from_electrons(
            electrons=1.0
        ) == pytest.approx(0.39346, 1.0e-4)
        assert trap.electrons_released_from_electrons(
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
        assert parameters.parallel_traps[0].exponential_factor == pytest.approx(
            0.6321, 1.0e-4
        )
        assert parameters.parallel_traps[1].density == 0.2
        assert parameters.parallel_traps[1].lifetime == 2.0
        assert parameters.parallel_traps[1].exponential_factor == pytest.approx(
            0.39346, 1.0e-4
        )

        assert type(parameters) == ac.ArcticParams
        assert type(parameters.serial_traps[0]) == ac.Trap
        assert type(parameters.serial_traps[1]) == ac.Trap

        assert parameters.serial_traps[0].density == 0.3
        assert parameters.serial_traps[0].lifetime == 3.0
        assert parameters.serial_traps[0].exponential_factor == pytest.approx(
            0.28346, 1.0e-4
        )
        assert parameters.serial_traps[1].density == 0.4
        assert parameters.serial_traps[1].lifetime == 4.0
        assert parameters.serial_traps[1].exponential_factor == pytest.approx(
            0.22119, 1.0e-4
        )

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


class TestInitialWatermarks:
    def test__initial_watermark_array__uses_rows_and_total_traps_to_set_size(
        self,
    ):
        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = trap_manager.initial_watermarks_from_rows_and_total_traps(
            rows=3, total_traps=1
        )

        assert (watermarks == np.zeros(shape=(3, 2))).all()

        watermarks = trap_manager.initial_watermarks_from_rows_and_total_traps(
            rows=3, total_traps=5
        )

        assert (watermarks == np.zeros(shape=(3, 6))).all()

        watermarks = trap_manager.initial_watermarks_from_rows_and_total_traps(
            rows=5, total_traps=2
        )

        assert (watermarks == np.zeros(shape=(5, 3))).all()


class TestElectronsReleasedAndUpdatedWatermarks:
    def test__empty_release(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        electrons_released = trap_manager.electrons_released_in_pixel()

        assert electrons_released == pytest.approx(0)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__single_trap(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )

        electrons_released = trap_manager.electrons_released_in_pixel()

        assert electrons_released == pytest.approx(2.5)
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]]
            )
        )

    def test__multiple_traps(self):

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [
                [0.5, 0.8, 0.3],
                [0.2, 0.4, 0.2],
                [0.1, 0.2, 0.1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        electrons_released = trap_manager.electrons_released_in_pixel()

        assert electrons_released == pytest.approx(2.5 + 1.28)
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [
                    [0.5, 0.4, 0.06],
                    [0.2, 0.2, 0.04],
                    [0.1, 0.1, 0.02],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )
        )


class TestElectronsCapturedByTraps:
    def test__first_capture(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        electron_fractional_height = 0.5

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(0.5 * 10)

    def test__new_highest_watermark(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.9

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(
            (0.1 + 0.12 + 0.07 + 0.1) * 10
        )

    def test__middle_new_watermarks(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.6

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx((0.1 + 0.06) * 10)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.75

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx((0.1 + 0.12 + 0.035) * 10)

    def test__new_lowest_watermark(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(0.3 * 0.2 * 10)

    def test__multiple_traps(self):

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

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

        electron_fractional_height = 0.6

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(
            (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8
        )


class TestUpdateWatermarks:
    def test__first_capture(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = trap_manager.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )
        electron_fractional_height = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_highest_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.9

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.9, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__middle_new_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.6

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.6, 1], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]])
        )

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.75

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.75, 1], [0.05, 0.3], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_lowest_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array(
                [[0.3, 1], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]]
            )
        )

    def test__multiple_traps(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [
                [0.5, 0.8, 0.7, 0.6],
                [0.2, 0.4, 0.3, 0.2],
                [0.1, 0.3, 0.2, 0.1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        electron_fractional_height = 0.6

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array(
                [
                    [0.6, 1, 1, 1],
                    [0.1, 0.4, 0.3, 0.2],
                    [0.1, 0.3, 0.2, 0.1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )


class TestUpdateWatermarksNotEnough:
    def test__first_captrue(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        electron_fractional_height = 0.5
        enough = 0.7

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            enough=enough,
        )
        print(watermarks)  ###
        assert watermarks == pytest.approx(
            np.array([[0.5, 0.7], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_highest_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]]
        )

        electron_fractional_height = 0.8
        enough = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )

        assert watermarks == pytest.approx(
            np.array(
                [[0.5, 0.9], [0.2, 0.7], [0.1, 0.5], [0, 0], [0, 0], [0, 0]]
            )
        )

    def test__new_middle_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )

        electron_fractional_height = 0.6
        enough = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )

        assert watermarks == pytest.approx(
            np.array(
                [[0.5, 0.9], [0.1, 0.7], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0]]
            )
        )

    def test__new_lowest_watermark(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3
        enough = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )

        assert watermarks == pytest.approx(
            np.array(
                [[0.3, 0.9], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]]
            )
        )

    def test__multiple_traps(self):

        trap_manager = ac.TrapManager(traps=[None], rows=6)

        watermarks = np.array(
            [
                [0.5, 0.8, 0.7, 0.6],
                [0.2, 0.4, 0.3, 0.2],
                [0.1, 0.3, 0.2, 0.1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        electron_fractional_height = 0.6
        enough = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )
        assert watermarks == pytest.approx(
            np.array(
                [
                    [0.5, 0.9, 0.85, 0.8],
                    [0.1, 0.7, 0.65, 0.6],
                    [0.1, 0.4, 0.3, 0.2],
                    [0.1, 0.3, 0.2, 0.1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )


class TestElectronsCapturedInPixel:
    def test__first_capture(self):

        electrons_available = 2500  # --> electron_fractional_height = 0.5

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured = trap_manager.electrons_captured_in_pixel(
            electrons_available=electrons_available, ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__multiple_traps(self):

        electrons_available = 3600  # --> electron_fractional_height = 0.6

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

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

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured = trap_manager.electrons_captured_in_pixel(
            electrons_available=electrons_available, ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(
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

        electrons_available = 2.5e-3  # --> electron_fractional_height = 4.9999e-4, enough=enough = 0.50001

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured = trap_manager.electrons_captured_in_pixel(
            electrons_available=electrons_available, ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(0.0025)
        assert trap_manager.watermarks == pytest.approx(
            np.array(
                [[4.9999e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        )

    def test__not_enough__multiple_traps(self):

        electrons_available = 4.839e-4  # --> electron_fractional_height = 2.199545e-4, enough=enough = 0.5

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

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

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured = trap_manager.electrons_captured_in_pixel(
            electrons_available=electrons_available, ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(
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
