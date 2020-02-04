import numpy as np
import pytest

import arctic as ac
from arctic import clock


class TestElectronsReleasedInPixel:
    def test__empty_release(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        clocker = ac.Clocker()
        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )

        electrons_released, watermarks = clock.electrons_released_in_pixel(
            watermarks=watermarks, traps=traps
        )

        assert electrons_released == pytest.approx(0)
        assert watermarks == pytest.approx(
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__single_trap(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
        )

        electrons_released, watermarks = clock.electrons_released_in_pixel(
            watermarks=watermarks, traps=traps
        )

        assert electrons_released == pytest.approx(2.5)
        assert watermarks == pytest.approx(
            np.array(
                [[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]]
            )
        )

    def test__multiple_traps(self):

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]

        watermarks = np.array(
            [
                [0.5, 0.8, 0.3],
                [0.2, 0.4, 0.2],
                [0.1, 0.2, 0.1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        electrons_released, watermarks = clock.electrons_released_in_pixel(
            watermarks=watermarks, traps=traps
        )

        assert electrons_released == pytest.approx(2.5 + 1.28)
        assert watermarks == pytest.approx(
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

        clocker = ac.Clocker()

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )
        electron_fractional_height = 0.5

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx(0.5 * 10)

    def test__new_highest_watermark(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.9

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx(
            (0.1 + 0.12 + 0.07 + 0.1) * 10
        )

    def test__middle_new_watermarks(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.6

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx((0.1 + 0.06) * 10)

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.75

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx((0.1 + 0.12 + 0.035) * 10)

    def test__new_lowest_watermark(self):

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx(0.3 * 0.2 * 10)

    def test__multiple_traps(self):

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]

        watermarks = np.array(
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

        electrons_captured = clock.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            traps=traps,
        )

        assert electrons_captured == pytest.approx(
            (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8
        )


class TestUpdateWatermarks:
    def test__first_capture(self):

        clocker = ac.Clocker()

        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )
        electron_fractional_height = 0.5

        watermarks = clock.update_watermarks(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_highest_watermark(self):

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.9

        watermarks = clock.update_watermarks(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.9, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__middle_new_watermark(self):

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.6

        watermarks = clock.update_watermarks(
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

        watermarks = clock.update_watermarks(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array([[0.75, 1], [0.05, 0.3], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_lowest_watermark(self):

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3

        watermarks = clock.update_watermarks(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )

        assert watermarks == pytest.approx(
            np.array(
                [[0.3, 1], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]]
            )
        )

    def test__multiple_traps(self):

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

        watermarks = clock.update_watermarks(
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

        clocker = ac.Clocker()

        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )

        electron_fractional_height = 0.5
        enough = 0.7

        watermarks = clock.update_watermarks_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )

        assert watermarks == pytest.approx(
            np.array([[0.5, 0.7], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__new_highest_watermark(self):

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]]
        )

        electron_fractional_height = 0.8
        enough = 0.5

        watermarks = clock.update_watermarks_not_enough(
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

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )

        electron_fractional_height = 0.6
        enough = 0.5

        watermarks = clock.update_watermarks_not_enough(
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

        watermarks = np.array(
            [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
        )
        electron_fractional_height = 0.3
        enough = 0.5

        watermarks = clock.update_watermarks_not_enough(
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

        watermarks = clock.update_watermarks_not_enough(
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

        clocker = ac.Clocker()
        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured, watermarks = clock.electrons_captured_in_pixel(
            electrons_available=electrons_available,
            watermarks=watermarks,
            traps=traps,
            ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(5)
        assert watermarks == pytest.approx(
            np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        )

    def test__multiple_traps(self):

        electrons_available = 3600  # --> electron_fractional_height = 0.6

        watermarks = np.array(
            [
                [0.5, 0.8, 0.7],
                [0.2, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured, watermarks = clock.electrons_captured_in_pixel(
            electrons_available=electrons_available,
            watermarks=watermarks,
            traps=traps,
            ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(
            (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8
        )
        assert watermarks == pytest.approx(
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

        clocker = ac.Clocker()
        watermarks = clocker.initial_watermarks_from_rows_and_total_traps(
            rows=6, total_traps=1
        )

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured, watermarks = clock.electrons_captured_in_pixel(
            electrons_available=electrons_available,
            watermarks=watermarks,
            traps=traps,
            ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(0.0025)
        assert watermarks == pytest.approx(
            np.array(
                [[4.9999e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        )

    def test__not_enough__multiple_traps(self):

        electrons_available = 4.839e-4  # --> electron_fractional_height = 2.199545e-4, enough=enough = 0.5

        watermarks = np.array(
            [
                [0.5, 0.8, 0.7],
                [0.2, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=8, lifetime=-1 / np.log(0.2)),
        ]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.5, well_max_height=10000, well_notch_depth=1e-7
        )

        electrons_captured, watermarks = clock.electrons_captured_in_pixel(
            electrons_available=electrons_available,
            watermarks=watermarks,
            traps=traps,
            ccd_volume=ccd_volume,
        )

        assert electrons_captured == pytest.approx(
            2.199545e-4 * (0.1 * 10 + 0.15 * 8)
        )
        assert watermarks == pytest.approx(
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


class TestClocker:
    def test__express_matrix_from_rows(self):
        clocker = ac.Clocker(express=1)

        express_multiplier = clocker.express_matrix_from_rows(rows=12)

        assert express_multiplier == pytest.approx(np.array([np.arange(1, 13)]))

        clocker = ac.Clocker(express=4)

        express_multiplier = clocker.express_matrix_from_rows(rows=12)

        assert express_multiplier == pytest.approx(
            np.array(
                [
                    [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                ]
            )
        )

        clocker = ac.Clocker(express=12)

        express_multiplier = clocker.express_matrix_from_rows(rows=12)

        assert express_multiplier == pytest.approx(np.triu(np.ones((12, 12))))

    def test__initial_watermark_array__uses_rows_and_total_traps_to_set_size(
        self,
    ):
        clock = ac.Clocker(sequence=None)

        watermarks = clock.initial_watermarks_from_rows_and_total_traps(
            rows=3, total_traps=1
        )

        assert (watermarks == np.zeros(shape=(3, 2))).all()

        watermarks = clock.initial_watermarks_from_rows_and_total_traps(
            rows=3, total_traps=5
        )

        assert (watermarks == np.zeros(shape=(3, 6))).all()

        watermarks = clock.initial_watermarks_from_rows_and_total_traps(
            rows=5, total_traps=2
        )

        assert (watermarks == np.zeros(shape=(5, 3))).all()

    def test__add_cti__single_pixel__compare_to_cpluspus_version(self):
        clocker = ac.Clocker(express=0)

        image = np.zeros((6, 2))
        image[2, 1] = 1000

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        image = clocker.add_cti(image=image, traps=traps, ccd_volume=ccd_volume)

        assert image == pytest.approx(
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 999.1396],
                    [0, 0.4292094],
                    [0, 0.2149715],
                    [0, 0.1077534],
                ]
            ),
            abs=1e-3,
        )

    def test__remove_cti__single_pixel__compare_to_cplusplus_version(self):
        image = np.zeros((6, 2))
        image[2, 1] = 1000

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        clocker = ac.Clocker(express=0)

        image_add = clocker.add_cti(
            image=image, traps=traps, ccd_volume=ccd_volume
        )

        # Check similarity after different iterations
        iterations_tolerance_dict = {
            1: 1e-2,
            2: 1e-5,
            3: 1e-7,
            4: 1e-10,
            5: 1e-12,
        }

        for iterations, tolerance in iterations_tolerance_dict.items():
            clocker = ac.Clocker(express=0, iterations=iterations)

            image_rem = clocker.remove_cti(
                image=image_add, traps=traps, ccd_volume=ccd_volume
            )

            assert image_rem == pytest.approx(image, abs=tolerance)


class TestAddCTI:
    def test__square__horizontal_line__line_loses_charge_trails_appear(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (
            image_difference[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        assert (
            image_difference[2, :] < 0.0
        ).all()  # All pixels in the charge line should lose charge due to capture
        assert (
            image_difference[3:-1, :] > 0.0
        ).all()  # All other pixels should have charge trailed into them

    def test__square__vertical_line__no_trails(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchanged
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (
            image_difference[:, 2] < 0.0
        ).all()  # charge line still loses charge

    def test__square__double_density__more_captures_so_brighter_trails(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        # SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED #

        trap_0 = ac.Trap(density=0.1, lifetime=1.0)
        trap_1 = ac.Trap(density=0.2, lifetime=1.0)

        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_0], ccd_volume=ccd_volume
        )
        image_post_cti_1 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_1], ccd_volume=ccd_volume
        )

        assert (
            image_post_cti_0[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        assert (
            image_post_cti_1[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        # noinspection PyUnresolvedReferences
        assert (
            image_post_cti_0[2, :] > image_post_cti_1[4, :]
        ).all()  # charge line loses less charge in image 1
        # noinspection PyUnresolvedReferences
        assert (image_post_cti_0[3:-1, :] < image_post_cti_1[5:-1, :]).all()
        # fewer pixels trailed behind in image 2

    def test__square__double_lifetime__longer_release_so_fainter_trails(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        # SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED #

        trap_0 = ac.Trap(density=0.1, lifetime=1.0)
        trap_1 = ac.Trap(density=0.1, lifetime=2.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_0], ccd_volume=ccd_volume
        )
        image_post_cti_1 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_1], ccd_volume=ccd_volume
        )

        assert (
            image_post_cti_0[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        assert (
            image_post_cti_1[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        # noinspection PyUnresolvedReferences
        assert (
            image_post_cti_0[2, :] == image_post_cti_1[2, :]
        ).all()  # charge line loses equal amount of charge
        # noinspection PyUnresolvedReferences
        assert (image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]).all()
        # each trail in pixel 2 is 'longer' so fainter

    def test__square__increase_beta__fewer_captures_fainter_trail(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        # SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED #

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_0 = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )
        ccd_1 = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.9, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_0
        )
        image_post_cti_1 = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_1
        )

        assert (
            image_post_cti_0[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        assert (
            image_post_cti_1[0:2, :] == 0.0
        ).all()  # First four rows should all remain zero
        # noinspection PyUnresolvedReferences
        assert (image_post_cti_0[2, :] < image_post_cti_1[2, :]).all()
        # charge line loses less charge with higher beta
        # noinspection PyUnresolvedReferences
        assert (
            image_post_cti_0[3:-1, :] > image_post_cti_1[3:-1, :]
        ).all()  # so less electrons trailed into image

    def test__square__two_traps_half_density_of_one__same_trails(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        # SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED #

        trap_0 = ac.Trap(density=0.1, lifetime=1.0)
        trap_1 = ac.Trap(density=0.05, lifetime=1.0)

        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_0], ccd_volume=ccd_volume
        )
        image_post_cti_1 = clocker.add_cti(
            image=image_pre_cti, traps=[trap_1, trap_1], ccd_volume=ccd_volume
        )

        # noinspection PyUnresolvedReferences
        assert (image_post_cti_0 == image_post_cti_1).all()

    def test__square__delta_functions__add_cti_only_behind_them(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (
            image_difference[0:3, 3] == 0.0
        ).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (
            image_difference[0:2, 4] == 0.0
        ).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

    def test__rectangle__horizontal_line__rectangular_image_odd_x_odd(self):
        image_pre_cti = np.zeros((5, 7))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0:2, :] == 0.0).all()
        assert (image_difference[2, :] < 0.0).all()
        assert (image_difference[3:-1, :] > 0.0).all()

    def test__rectangle__horizontal_line__rectangular_image_even_x_even(self):
        image_pre_cti = np.zeros((4, 6))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0:2, :] == 0.0).all()
        assert (image_difference[2, :] < 0.0).all()
        assert (image_difference[3:-1, :] > 0.0).all()

    def test__rectangle__horizontal_line__rectangular_image_even_x_odd(self):
        image_pre_cti = np.zeros((4, 7))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0:2, :] == 0.0).all()
        assert (image_difference[2, :] < 0.0).all()
        assert (image_difference[3:-1, :] > 0.0).all()

    def test__rectangle__horizontal_line__rectangular_image_odd_x_even(self):
        image_pre_cti = np.zeros((5, 6))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )
        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0:2, :] == 0.0).all()
        assert (image_difference[2, :] < 0.0).all()
        assert (image_difference[3:-1, :] > 0.0).all()

    def test__rectangle__vertical_line__rectangular_image_odd_x_odd(self):
        image_pre_cti = np.zeros((3, 5))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()

    def test__rectangle__vertical_line__rectangular_image_even_x_even(self):
        image_pre_cti = np.zeros((4, 6))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()

    def test__rectangle__vertical_line__rectangular_image_even_x_odd(self):
        image_pre_cti = np.zeros((4, 7))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()

    def test__rectangle__vertical_line__rectangular_image_odd_x_even(self):
        image_pre_cti = np.zeros((5, 6))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()

    def test__rectangle__delta_functions__add_cti_only_behind_them__odd_x_odd(
        self,
    ):
        image_pre_cti = np.zeros((5, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (
            image_difference[0:3, 3] == 0.0
        ).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (
            image_difference[0:2, 4] == 0.0
        ).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__even_x_even(
        self,
    ):
        image_pre_cti = np.zeros((6, 8))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (
            image_difference[0:3, 3] == 0.0
        ).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (
            image_difference[0:2, 4] == 0.0
        ).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 7] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__even_x_odd(
        self,
    ):
        image_pre_cti = np.zeros((6, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (
            image_difference[0:3, 3] == 0.0
        ).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (
            image_difference[0:2, 4] == 0.0
        ).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__odd_x_even(
        self,
    ):
        image_pre_cti = np.zeros((5, 6))

        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (
            image_difference[0:3, 3] == 0.0
        ).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (
            image_difference[0:2, 4] == 0.0
        ).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge


class TestArcticCorrectCTI:
    def test__square__horizontal_line__corrected_image_more_like_original(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__square__vertical_line__corrected_image_more_like_original(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__square__delta_functions__corrected_image_more_like_original(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__square__decrease_iterations__worse_correction(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker_x5 = ac.Clocker(iterations=5, express=0)

        image_post_cti = clocker_x5.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_correct_cti = clocker_x5.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_niter_5 = image_correct_cti - image_pre_cti

        clocker_x3 = ac.Clocker(iterations=3, express=0)

        image_correct_cti = clocker_x3.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_niter_3 = image_correct_cti - image_pre_cti

        # noinspection PyUnresolvedReferences
        assert (
            abs(image_difference_niter_5) <= abs(image_difference_niter_3)
        ).all()  # First four rows should all remain zero

    def test__rectangle__horizontal_line__odd_x_odd(self):
        image_pre_cti = np.zeros((5, 3))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__horizontal_line__even_x_even(self):
        image_pre_cti = np.zeros((6, 4))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__horizontal_line__even_x_odd(self):
        image_pre_cti = np.zeros((6, 3))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__horizontal_line__odd_x_even(self):
        image_pre_cti = np.zeros((5, 4))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__veritcal_line__odd_x_odd(self):
        image_pre_cti = np.zeros((5, 3))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__veritcal_line__even_x_even(self):
        image_pre_cti = np.zeros((6, 4))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__veritcal_line__even_x_odd(self):
        image_pre_cti = np.zeros((6, 3))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__veritcal_line__odd_x_even(self):
        image_pre_cti = np.zeros((5, 4))
        image_pre_cti[:, 2] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__delta_functions__odd_x_odd(self):
        image_pre_cti = np.zeros((5, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__delta_functions__even_x_even(self):
        image_pre_cti = np.zeros((6, 8))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__delta_functions__even_x_odd(self):
        image_pre_cti = np.zeros((6, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero

    def test__rectangle__delta_functions__odd_x_even(self):
        image_pre_cti = np.zeros((5, 8))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        clocker = ac.Clocker(iterations=1, express=0)

        image_post_cti = clocker.add_cti(
            image=image_pre_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = clocker.remove_cti(
            image=image_post_cti, traps=[trap], ccd_volume=ccd_volume
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero
