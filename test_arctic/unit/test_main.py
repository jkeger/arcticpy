import numpy as np
import pytest

import arctic as ac


class TestGeneral:
    def test__express_matrix_from_rows(self):

        express_multiplier = ac.express_matrix_from_rows_and_express(rows=12, express=1)

        assert express_multiplier == pytest.approx(np.array([np.arange(1, 13)]))

        express_multiplier = ac.express_matrix_from_rows_and_express(rows=12, express=4)

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

        express_multiplier = ac.express_matrix_from_rows_and_express(
            rows=12, express=12
        )

        assert express_multiplier == pytest.approx(np.triu(np.ones((12, 12))))

        for rows in [5, 11, 199, 360]:
            for express in [0, 1, 2, 3, 7, 199]:
                for offset in [0, 1, 13]:
                    for integer_express_multiplier in [True, False]:
                        express_multiplier = ac.express_matrix_from_rows_and_express(
                            rows=rows,
                            express=express,
                            offset=offset,
                            integer_express_multiplier=integer_express_multiplier,
                        )
                        assert np.sum(express_multiplier, axis=0) == pytest.approx(
                            np.arange(1, rows + 1) + offset
                        )

    def test__add_cti__parallel_only__single_pixel__compare_to_cpluspus_version(self,):
        image = np.zeros((6, 2))
        image[2, 1] = 1000

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        image = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
        )

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

    def test__remove_cti__parallel_only__single_pixel__successive_iterations(self,):
        image = np.zeros((6, 2))
        image[2, 1] = 1000

        traps = [ac.Trap(density=10, lifetime=-1 / np.log(0.5))]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        image_add = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
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
            image_rem = ac.remove_cti(
                image=image_add,
                iterations=iterations,
                parallel_traps=traps,
                parallel_ccd_volume=ccd_volume,
            )

            assert image_rem == pytest.approx(image, abs=tolerance)

    def test__multiple_trap_managers(self):
        image = np.zeros((6, 2))
        image[2, 1] = 1000

        # Single trap manager
        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=5, lifetime=-1 / np.log(0.5)),
        ]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        image_single = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
        )

        # Multiple trap managers
        traps = [
            [ac.Trap(density=10, lifetime=-1 / np.log(0.5))],
            [ac.Trap(density=5, lifetime=-1 / np.log(0.5))],
        ]

        image_multi = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
        )

        assert (image_single == image_multi).all

    def test__different_trap_managers(self):
        image = np.zeros((6, 2))
        image[2, 1] = 1000

        # Standard trap manager
        traps = [
            ac.Trap(density=10, lifetime=-1 / np.log(0.5)),
            ac.Trap(density=5, lifetime=-1 / np.log(0.5)),
        ]

        ccd_volume = ac.CCDVolume(
            well_fill_beta=0.8, well_max_height=8.47e4, well_notch_depth=1e-7
        )

        image_std = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
        )

        # Multiple trap managers, non-uniform distribution behaving like normal
        traps = [
            [ac.Trap(density=10, lifetime=-1 / np.log(0.5))],
            [
                ac.TrapNonUniformHeightDistribution(
                    density=5,
                    lifetime=-1 / np.log(0.5),
                    electron_fractional_height_min=0,
                    electron_fractional_height_max=1,
                )
            ],
        ]

        image_multi = ac.add_cti(
            image=image, parallel_traps=traps, parallel_ccd_volume=ccd_volume
        )

        assert (image_std == image_multi).all


class TestAddCTIParallelOnly:
    def test__square__horizontal_line__line_loses_charge_trails_appear(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()  # Most pixels unchanged
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()  # charge line still loses charge

    def test__square__double_density__more_captures_so_brighter_trails(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        # SETUP TWO SETS OF PARAMETERS WITH ONE PARAMETER DOUBLED #

        trap_0 = ac.Trap(density=0.1, lifetime=1.0)
        trap_1 = ac.Trap(density=0.2, lifetime=1.0)

        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_0],
            parallel_ccd_volume=ccd_volume,
        )
        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_1],
            parallel_ccd_volume=ccd_volume,
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

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_0],
            parallel_ccd_volume=ccd_volume,
        )
        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_1],
            parallel_ccd_volume=ccd_volume,
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

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_0,
        )
        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_1,
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

        # NOW GENERATE THE IMAGE POST CTI OF EACH SET

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_0],
            parallel_ccd_volume=ccd_volume,
        )
        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_traps=[trap_1, trap_1],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

    def test__rectangle__horizontal_line__rectangular_image_odd_x_odd(self):
        image_pre_cti = np.zeros((5, 7))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0:2] == 0.0).all()
        assert (image_difference[:, 3:-1] == 0.0).all()
        assert (image_difference[:, 2] < 0.0).all()

    def test__rectangle__delta_functions__add_cti_only_behind_them__odd_x_odd(self,):
        image_pre_cti = np.zeros((5, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__even_x_even(self,):
        image_pre_cti = np.zeros((6, 8))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 7] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__even_x_odd(self,):
        image_pre_cti = np.zeros((6, 7))
        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge
        assert (image_difference[:, 6] == 0.0).all()  # No Delta, no charge

    def test__rectangle__delta_functions__add_cti_only_behind_them__odd_x_even(self,):
        image_pre_cti = np.zeros((5, 6))

        image_pre_cti[1, 1] += 100  # Delta 1
        image_pre_cti[3, 3] += 100  # Delta 2
        image_pre_cti[2, 4] += 100  # Delta 3

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[:, 0] == 0.0).all()  # No Delta, no charge

        assert image_difference[0, 1] == 0.0  # No charge in front of Delta 1
        assert image_difference[1, 1] < 0.0  # Delta 1 loses charge
        assert (image_difference[2:5, 1] > 0.0).all()  # Delta 1 trails

        assert (image_difference[:, 2] == 0.0).all()  # No Delta, no charge

        assert (image_difference[0:3, 3] == 0.0).all()  # No charge in front of Delta 2
        assert image_difference[3, 3] < 0.0  # Delta 2 loses charge
        assert image_difference[4, 3] > 0.0  # Delta 2 trail

        assert (image_difference[0:2, 4] == 0.0).all()  # No charge in front of Delta 3
        assert image_difference[2, 4] < 0.0  # Delta 3 loses charge
        assert (image_difference[3:5, 4] > 0.0).all()  # Delta 3 trail

        assert (image_difference[:, 5] == 0.0).all()  # No Delta, no charge


class TestArcticAddCTIParallelAndSerial:
    def test__horizontal_charge_line__loses_charge_trails_form_both_directions(self,):

        parallel_traps = [ac.Trap(density=0.4, lifetime=1.0)]
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        serial_traps = [ac.Trap(density=0.2, lifetime=2.0)]
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.4, well_max_height=84700
        )

        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, 1:4] = +100

        image_post_cti = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0:2, :] == 0.0).all()  # No change in front of charge
        assert (image_difference[2, 1:4] < 0.0).all()  # charge lost in charge

        assert (image_difference[3:5, 1:4] > 0.0).all()  # Parallel trails behind charge

        assert image_difference[2:5, 0] == pytest.approx(0.0, 1.0e-4)
        # no serial cti trail to left
        assert (
            image_difference[2:5, 4] > 0.0
        ).all()  # serial cti trail to right including parallel cti trails

        assert (image_difference[3, 1:4] > image_difference[4, 1:4]).all()
        # check parallel cti trails decreasing.

        assert (
            image_difference[3, 4] > image_difference[4, 4]
        )  # Check serial trails of parallel trails decreasing.

    def test__vertical_charge_line__loses_charge_trails_form_in_serial_directions(
        self,
    ):

        parallel_traps = [ac.Trap(density=0.4, lifetime=1.0)]
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.000001, well_fill_beta=0.8, well_max_height=84700
        )

        serial_traps = [ac.Trap(density=0.2, lifetime=2.0)]
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.000001, well_fill_beta=0.4, well_max_height=84700
        )

        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[1:4, 2] = +100

        image_post_cti = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert (image_difference[0, 0:5] == 0.0).all()  # No change in front of charge
        assert (image_difference[1:4, 2] < 0.0).all()  # charge lost in charge

        assert (image_difference[4, 2] > 0.0).all()  # Parallel trail behind charge

        assert image_difference[0:5, 0:2] == pytest.approx(0.0, 1.0e-4)
        assert (
            image_difference[1:5, 3:5] > 0.0
        ).all()  # serial cti trail to right including parallel cti trails

        assert (
            image_difference[3, 3] > image_difference[3, 4]
        )  # Check serial trails decreasing.
        assert (
            image_difference[4, 3] > image_difference[4, 4]
        )  # Check serial trails of parallel trails decreasing.

    def test__individual_pixel_trails_form_cross_around_it(self,):

        parallel_traps = [ac.Trap(density=0.4, lifetime=1.0)]
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        serial_traps = [ac.Trap(density=0.2, lifetime=2.0)]
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.4, well_max_height=84700
        )

        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, 2] = +100

        image_post_cti = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti - image_pre_cti

        assert image_difference[0:2, :] == pytest.approx(
            0.0, 1.0e-4
        )  # First two rows should all remain zero
        assert image_difference[:, 0:2] == pytest.approx(
            0.0, 1.0e-4
        )  # First tow columns should all remain zero
        assert (
            image_difference[2, 2] < 0.0
        )  # pixel which had charge should lose it due to cti.
        assert (
            image_difference[3:5, 2] > 0.0
        ).all()  # Parallel trail increases charge above pixel
        assert (
            image_difference[2, 3:5] > 0.0
        ).all()  # Serial trail increases charge to right of pixel
        assert (
            image_difference[3:5, 3:5] > 0.0
        ).all()  # Serial trailing of parallel trail increases charge up-right of pixel

    def test__individual_pixel_double_density__more_captures_so_brighter_trails(self,):

        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, 2] = +100

        parallel_traps = [ac.Trap(density=0.4, lifetime=1.0)]
        serial_traps = [ac.Trap(density=0.2, lifetime=2.0)]

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        parallel_traps = [ac.Trap(density=0.8, lifetime=1.0)]
        serial_traps = [ac.Trap(density=0.4, lifetime=2.0)]

        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti_1 - image_post_cti_0

        assert image_difference[0:2, :] == pytest.approx(
            0.0, 1.0e-4
        )  # First two rows remain zero
        assert image_difference[:, 0:2] == pytest.approx(
            0.0, 1.0e-4
        )  # First tow columns remain zero
        assert (
            image_difference[2, 2] < 0.0
        )  # More captures in second ci_pre_ctis, so more charge in central pixel lost
        assert (
            image_difference[3:5, 2] > 0.0
        ).all()  # More captpures in ci_pre_ctis 2, so brighter parallel trail
        assert (
            image_difference[2, 3:5] > 0.0
        ).all()  # More captpures in ci_pre_ctis 2, so brighter serial trail
        assert (
            image_difference[3:5, 3:5] > 0.0
        ).all()  # Brighter serial trails from parallel trail trails

    def test__individual_pixel_increase_lifetime_longer_release_so_fainter_trails(
        self,
    ):

        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, 2] = +100

        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        parallel_traps = [ac.Trap(density=0.1, lifetime=1.0)]
        serial_traps = [ac.Trap(density=0.1, lifetime=1.0)]

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        parallel_traps = [ac.Trap(density=0.1, lifetime=20.0)]
        serial_traps = [ac.Trap(density=0.1, lifetime=20.0)]

        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti_1 - image_post_cti_0

        assert image_difference[0:2, :] == pytest.approx(
            0.0, 1.0e-4
        )  # First two rows remain zero
        assert image_difference[:, 0:2] == pytest.approx(
            0.0, 1.0e-4
        )  # First tow columns remain zero
        assert image_difference[2, 2] == pytest.approx(
            0.0, 1.0e-4
        )  # Same density so same captures
        assert (
            image_difference[3:5, 2] < 0.0
        ).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail
        assert (
            image_difference[2, 3:5] < 0.0
        ).all()  # Longer release in ci_pre_ctis 2, so fainter serial trail
        assert (
            image_difference[3:5, 3:5] < 0.0
        ).all()  # Longer release in ci_pre_ctis 2, so fainter parallel trail trails

    def test__individual_pixel_increase_beta__fewer_captures_so_fainter_trails(self,):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, 2] = +100

        parallel_traps = [ac.Trap(density=0.1, lifetime=1.0)]
        serial_traps = [ac.Trap(density=0.1, lifetime=1.0)]

        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti_0 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.9, well_max_height=84700
        )
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.9, well_max_height=84700
        )

        image_post_cti_1 = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference = image_post_cti_1 - image_post_cti_0

        assert image_difference[0:2, :] == pytest.approx(
            0.0, 1.0e-4
        )  # First two rows remain zero
        assert image_difference[:, 0:2] == pytest.approx(
            0.0, 1.0e-4
        )  # First tow columns remain zero
        assert image_difference[2, 2] > 0.0  # Higher beta in 2, so fewer captures
        assert (
            image_difference[3:5, 2] < 0.0
        ).all()  # Fewer catprues in 2, so fainter parallel trail
        assert (
            image_difference[2, 3:5] < 0.0
        ).all()  # Fewer captures in 2, so fainter serial trail
        assert (
            image_difference[3:5, 3:5] < 0.0
        ).all()  # fewer captures in 2, so fainter trails trail region


class TestAddCTIParallelMultiPhase:
    def test__square__horizontal_line__line_loses_charge_trails_appear(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01,
            well_fill_beta=0.8,
            well_max_height=84700,
            phase_widths=[0.5, 0.2, 0.2, 0.1],
        )

        clocker = ac.Clocker(sequence=[0.5, 0.2, 0.2, 0.1])

        image_post_cti = ac.add_cti(
            image=image_pre_cti,
            parallel_clocker=clocker,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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


class TestArcticCorrectCTIParallelOnly:
    def test__square__horizontal_line__corrected_image_more_like_original(self):
        image_pre_cti = np.zeros((5, 5))
        image_pre_cti[2, :] += 100

        trap = ac.Trap(density=0.1, lifetime=1.0)
        ccd_volume = ac.CCDVolume(
            well_notch_depth=0.01, well_fill_beta=0.8, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=5,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
        )

        image_difference_niter_5 = image_correct_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=3,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
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

        image_post_cti = ac.add_cti(
            image=image_pre_cti, parallel_traps=[trap], parallel_ccd_volume=ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_post_cti,
            iterations=1,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
        )

        image_difference_2 = image_correct_cti - image_pre_cti

        assert (
            image_difference_2 <= abs(image_difference_1)
        ).all()  # First four rows should all remain zero


class TestArcticCorrectCTIParallelAndSerial:
    def test__array_of_values__corrected_image_more_like_original(self,):
        image_pre_cti = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 5.0, 9.5, 3.2, 9.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 5.0, 9.5, 352, 9.4, 0.0],
                [0.0, 9.0, 5.0, 9.5, 0.0, 9.0, 0.0],
                [0.0, 9.0, 9.1, 9.3, 9.2, 9.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        parallel_traps = [ac.Trap(density=0.4, lifetime=1.0)]
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.8, well_max_height=84700
        )

        serial_traps = [ac.Trap(density=0.2, lifetime=2.0)]
        serial_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.00001, well_fill_beta=0.4, well_max_height=84700
        )

        image_post_cti = ac.add_cti(
            image=image_pre_cti,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )

        image_difference_1 = image_post_cti - image_pre_cti

        image_correct_cti = ac.remove_cti(
            image=image_pre_cti,
            iterations=1,
            parallel_express=5,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_express=5,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
        )
        image_difference_2 = image_correct_cti - image_pre_cti

        assert (abs(image_difference_2) <= abs(image_difference_1)).all()
