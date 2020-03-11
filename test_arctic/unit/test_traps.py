import numpy as np
import pytest
from scipy import integrate

import arctic as ac


class TestInitialWatermarks:
    def test__initial_watermark_array__uses_rows_and_total_traps_to_set_size(self,):
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
            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]])
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

    def test__single_trap__change_time(self):

        # Half the time, half the liftime --> same result
        traps = [ac.Trap(density=10, lifetime=-0.5 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )

        electrons_released = trap_manager.electrons_released_in_pixel(time=0.5)

        assert electrons_released == pytest.approx(2.5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]])
        )

    def test__single_trap__change_width(self):

        # Half the time, double the density --> same result
        traps = [ac.Trap(density=20, lifetime=-1 / np.log(0.5))]
        trap_manager = ac.TrapManager(traps=traps, rows=6)

        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )

        electrons_released = trap_manager.electrons_released_in_pixel(width=0.5)

        assert electrons_released == pytest.approx(2.5)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]])
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

        assert electrons_captured == pytest.approx((0.1 + 0.12 + 0.07 + 0.1) * 10)

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
            np.array([[0.3, 1], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]])
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
    def test__first_capture(self):

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

        watermarks = np.array([[0.5, 0.8], [0.2, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]])

        electron_fractional_height = 0.8
        enough = 0.5

        watermarks = trap_manager.updated_watermarks_from_capture_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )

        assert watermarks == pytest.approx(
            np.array([[0.5, 0.9], [0.2, 0.7], [0.1, 0.5], [0, 0], [0, 0], [0, 0]])
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
            np.array([[0.5, 0.9], [0.1, 0.7], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0]])
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
            np.array([[0.3, 0.9], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]])
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
            electrons_available=electrons_available, ccd_volume=ccd_volume
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
            electrons_available=electrons_available, ccd_volume=ccd_volume
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
            electrons_available=electrons_available, ccd_volume=ccd_volume
        )

        assert electrons_captured == pytest.approx(0.0025)
        assert trap_manager.watermarks == pytest.approx(
            np.array([[4.9999e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
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
            electrons_available=electrons_available, ccd_volume=ccd_volume
        )

        assert electrons_captured == pytest.approx(2.199545e-4 * (0.1 * 10 + 0.15 * 8))
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


class TestTrapManagerNonUniformHeightDistribution:
    def test__effective_non_uniform_electron_fractional_height(self):

        traps = [
            ac.TrapNonUniformHeightDistribution(
                density=10,
                lifetime=-1 / np.log(0.5),
                electron_fractional_height_min=0.95,
                electron_fractional_height_max=1,
            )
        ]
        trap_manager = ac.TrapManagerNonUniformHeightDistribution(traps=traps, rows=6,)

        assert trap_manager.effective_non_uniform_electron_fractional_height(0.9) == 0
        assert trap_manager.effective_non_uniform_electron_fractional_height(1) == 1
        assert (
            trap_manager.effective_non_uniform_electron_fractional_height(0.975) == 0.5
        )

    def test__first_capture(self):

        traps = [
            ac.TrapNonUniformHeightDistribution(
                density=10,
                lifetime=-1 / np.log(0.5),
                electron_fractional_height_min=0.95,
                electron_fractional_height_max=1,
            )
        ]
        trap_manager = ac.TrapManagerNonUniformHeightDistribution(traps=traps, rows=6,)

        electron_fractional_height = 0.5

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(0.5 * 10)

        trap_manager = ac.TrapManagerNonUniformHeightDistribution(traps=traps, rows=6,)

        electron_fractional_height = 0.99

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx(0.99 * 10)

    def test__middle_new_watermarks(self):

        traps = [
            ac.TrapNonUniformHeightDistribution(
                density=10,
                lifetime=-1 / np.log(0.5),
                electron_fractional_height_min=0.95,
                electron_fractional_height_max=1,
            )
        ]
        trap_manager = ac.TrapManagerNonUniformHeightDistribution(traps=traps, rows=6,)

        trap_manager.watermarks = np.array(
            [[0.96, 0.8], [0.98, 0.4], [0.99, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.97

        electrons_captured = trap_manager.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager.watermarks,
            traps=trap_manager.traps,
        )

        assert electrons_captured == pytest.approx((0.96 * 0.2 + 0.01 * 0.6) * 10)


class TestTrapManagerTrackTime:
    def test__fill_fraction_from_time_elapsed(self):

        trap = ac.Trap(density=10, lifetime=2)

        fill = trap.fill_fraction_from_time_elapsed(1)
        assert fill == np.exp(-0.5)

        time = trap.time_elapsed_from_fill_fraction(0.5)
        assert time == -2 * np.log(0.5)

        assert fill == trap.fill_fraction_from_time_elapsed(trap.time_elapsed_from_fill_fraction(fill))
        assert time == trap.time_elapsed_from_fill_fraction(trap.fill_fraction_from_time_elapsed(time))

    def test__electrons_released_in_pixel_using_time(self):

        trap = ac.Trap(density=10, lifetime=-1 / np.log(0.5))
        trap_manager_fill = ac.TrapManager(traps=[trap], rows=6)
        trap_manager_time = ac.TrapManagerTrackTime(traps=[trap], rows=6)

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

        electrons_released_fill = trap_manager_fill.electrons_released_in_pixel()
        electrons_released_time = trap_manager_time.electrons_released_in_pixel()

        assert electrons_released_fill == electrons_released_time

    def test__electrons_captured_by_traps_using_time(self):

        trap = ac.Trap(density=10, lifetime=-1 / np.log(0.5))
        trap_manager_fill = ac.TrapManager(traps=[trap], rows=6)
        trap_manager_time = ac.TrapManagerTrackTime(traps=[trap], rows=6)

        trap_manager_fill.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        trap_manager_time.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.3)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        electron_fractional_height = 0.6

        electrons_captured_fill = trap_manager_fill.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager_fill.watermarks,
            traps=trap_manager_fill.traps,
        )
        electrons_captured_time = trap_manager_time.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=trap_manager_time.watermarks,
            traps=trap_manager_time.traps,
        )

        assert electrons_captured_fill == electrons_captured_time

    def test__updated_watermarks_from_capture_using_time(self):

        trap = ac.Trap(density=10, lifetime=-1 / np.log(0.5))
        trap_manager = ac.TrapManagerTrackTime(traps=[trap], rows=6)

        watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.3)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        electron_fractional_height = 0.75

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array(
                [
                    [0.75, 0],
                    [0.05, trap.time_elapsed_from_fill_fraction(0.3)],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            )
        )

    def test__updated_watermarks_from_capture_not_enough_multiple_traps_using_time(self):
        
        trap_1 = ac.Trap(density=10, lifetime=-1 / np.log(0.5))
        trap_2 = ac.Trap(density=10, lifetime=-2 / np.log(0.5))
        trap_manager = ac.TrapManagerTrackTime(traps=[trap_1, trap_2], rows=6)

        watermarks = np.array(
            [
                [0.5, trap_1.time_elapsed_from_fill_fraction(0.8), trap_2.time_elapsed_from_fill_fraction(0.6)],
                [0.2, trap_1.time_elapsed_from_fill_fraction(0.4), trap_2.time_elapsed_from_fill_fraction(0.2)],
                [0.1, trap_1.time_elapsed_from_fill_fraction(0.3), trap_2.time_elapsed_from_fill_fraction(0.1)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
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
                    [0.5, trap_1.time_elapsed_from_fill_fraction(0.9), trap_2.time_elapsed_from_fill_fraction(0.8)], 
                    [0.1, trap_1.time_elapsed_from_fill_fraction(0.7), trap_2.time_elapsed_from_fill_fraction(0.6)], 
                    [0.1, trap_1.time_elapsed_from_fill_fraction(0.4), trap_2.time_elapsed_from_fill_fraction(0.2)], 
                    [0.1, trap_1.time_elapsed_from_fill_fraction(0.3), trap_2.time_elapsed_from_fill_fraction(0.1)], 
                    [0, 0, 0], 
                    [0, 0, 0]
                ]
            )
        )


class TestTrapLifetimeContinuum:
    def test__distribution_of_traps_with_lifetime(self):
        
        median_lifetime = -1 / np.log(0.5)
        scale_lifetime = 0.5
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )

        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )
                
        # Check that the integral from zero to infinity is one
        assert integrate.quad(
            trap.distribution_of_traps_with_lifetime, 0, np.inf, args=(
                trap.middle_lifetime, trap.scale_lifetime
            )
        )[0] == pytest.approx(1)

    def test__fill_fraction_from_time_elapsed_narrow_continuum(self):

        # Check that narrow distribution gives similar result to single lifetime  
        # Simple trap
        trap = ac.Trap(density=10, lifetime=1)
        fill_single = trap.fill_fraction_from_time_elapsed(1)
        
        # Narrow continuum
        median_lifetime = 1
        scale_lifetime = 0.1
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )
        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )
        fill_continuum = trap.fill_fraction_from_time_elapsed(1)
        
        assert fill_continuum == pytest.approx(fill_single, rel=0.01)

    def test__time_elapsed_from_fill_fraction_narrow_continuum(self):

        # Check that narrow distribution gives similar result to single lifetime  
        # Simple trap
        trap = ac.Trap(density=10, lifetime=1)
        time_single = trap.time_elapsed_from_fill_fraction(0.5)
        
        # Narrow continuum
        median_lifetime = 1
        scale_lifetime = 0.1
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )
        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )
        time_continuum = trap.time_elapsed_from_fill_fraction(0.5)
        
        assert time_continuum == pytest.approx(time_single, rel=0.01)
        
    def test__time_elapsed_from_fill_fraction_continuum(self):
        
        median_lifetime = 1
        scale_lifetime = 1
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )
        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )
        time = 1
        
        assert trap.time_elapsed_from_fill_fraction(
            trap.fill_fraction_from_time_elapsed(time)
        ) == time
       
    def test__electrons_released_from_time_elapsed_and_time_narrow_continuum(self):
    
        # Check that narrow distribution gives similar result to single lifetime  
        # Simple trap
        trap = ac.Trap(density=10, lifetime=1)
        trap_manager = ac.TrapManager(traps=[trap], rows=6)
        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        electrons_released_single = trap_manager.electrons_released_in_pixel()
    
        # Narrow continuum
        median_lifetime = 1
        scale_lifetime = 0.1
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )
        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )        
        trap_manager = ac.TrapManagerTrackTime(traps=[trap], rows=6)
        trap_manager.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )        
        electrons_released_continuum = trap_manager.electrons_released_in_pixel()
    
        assert electrons_released_continuum == pytest.approx(
            electrons_released_single, rel=0.01
        )
    
    def test__electrons_released_from_time_elapsed_and_time_continuum(self):
    
        # Check that distribution gives somewhat similar result to single lifetime  
        # Simple trap
        trap = ac.Trap(density=10, lifetime=1)
        trap_manager = ac.TrapManager(traps=[trap], rows=6)
        trap_manager.watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )
        electrons_released_single = trap_manager.electrons_released_in_pixel()
    
        # Continuum
        median_lifetime = 1
        scale_lifetime = 1
        def trap_distribution(lifetime, median, scale):
            return np.exp(-(np.log(lifetime) - np.log(median))**2 / (2 * scale**2)) / (
                lifetime * scale * np.sqrt(2*np.pi)
            )
        trap = ac.TrapLifetimeContinuum(
            density=10, 
            distribution_of_traps_with_lifetime=trap_distribution,
            middle_lifetime=median_lifetime,
            scale_lifetime=scale_lifetime,
        )        
        trap_manager = ac.TrapManagerTrackTime(traps=[trap], rows=6)
        trap_manager.watermarks = np.array(
            [
                [0.5, trap.time_elapsed_from_fill_fraction(0.8)],
                [0.2, trap.time_elapsed_from_fill_fraction(0.4)],
                [0.1, trap.time_elapsed_from_fill_fraction(0.2)],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )        
        electrons_released_continuum = trap_manager.electrons_released_in_pixel()
    
        assert electrons_released_continuum == pytest.approx(
            electrons_released_single, rel=1
        )
    