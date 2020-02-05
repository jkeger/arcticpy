import numpy as np
import pytest

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
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
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
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
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
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
        )

        assert watermarks == pytest.approx(
            np.array([[0.6, 1], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]])
        )

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.75

        watermarks = trap_manager.updated_watermarks_from_capture(
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
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
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
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
            electron_fractional_height=electron_fractional_height, watermarks=watermarks
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

        electrons_available = (
            2.5e-3
        )  # --> electron_fractional_height = 4.9999e-4, enough=enough = 0.50001

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

        electrons_available = (
            4.839e-4
        )  # --> electron_fractional_height = 2.199545e-4, enough=enough = 0.5

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
