""" CTI python tests

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk
    
    WIP...
"""
from cti import *
import pytest


# //////////////////////////////////////////////////////////////////////////// #
#                               Test objects                                   #
# //////////////////////////////////////////////////////////////////////////// #
test_ccd_1 = CCD(
    well_fill_alpha=None,
    well_fill_beta=0.5,
    well_fill_gamma=None,
    well_max_height=10000,
    well_notch_height=1e-7,
)

test_species_1 = TrapSpecies(density=10, lifetime=-1 / np.log(0.5))
test_species_2 = TrapSpecies(density=8, lifetime=-1 / np.log(0.2))


# //////////////////////////////////////////////////////////////////////////// #
#                               Test Functions                                 #
# //////////////////////////////////////////////////////////////////////////// #
def test_create_express_multiplier():
    """ Test create_express_multiplier(). """

    # ========
    # Full express
    # ========
    A2_express_multiplier = create_express_multiplier(express=1, num_row=12)

    assert A2_express_multiplier == pytest.approx(np.array([np.arange(1, 13)]))

    # ========
    # Medium express
    # ========
    A2_express_multiplier = create_express_multiplier(express=4, num_row=12)

    assert A2_express_multiplier == pytest.approx(
        np.array(
            [
                [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
            ],
        )
    )

    # ========
    # No express
    # ========
    A2_express_multiplier = create_express_multiplier(express=12, num_row=12)

    assert A2_express_multiplier == pytest.approx(np.triu(np.ones((12, 12))))

    return 0


def test_release_electrons_in_pixel():
    """ Test release_electrons_in_pixel(). """

    # ========
    # Empty release
    # ========
    A1_trap_species = [test_species_1]
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )

    e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
        A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_rele == pytest.approx(0)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # Single trap species
    # ========
    A1_trap_species = [test_species_1]
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
    )

    e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
        A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_rele == pytest.approx(2.5)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # Multiple trap species
    # ========
    A1_trap_species = [test_species_1, test_species_2]
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.3],
            [0.2, 0.4, 0.2],
            [0.1, 0.2, 0.1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
        A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_rele == pytest.approx(2.5 + 1.28)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [
                [0.5, 0.4, 0.06],
                [0.2, 0.2, 0.04],
                [0.1, 0.1, 0.02],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
    )

    return 0


def test_capture_electrons():
    """ Test capture_electrons(). """

    # ========
    # First capture
    # ========
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    height_e = 0.5

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx(0.5 * 10)

    # ========
    # New highest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.9

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx((0.1 + 0.12 + 0.07 + 0.1) * 10)

    # ========
    # Middle new watermark (a)
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.6

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx((0.1 + 0.06) * 10)

    # ========
    # Middle new watermark (b)
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.75

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx((0.1 + 0.12 + 0.035) * 10)

    # ========
    # New lowest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.3

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx(0.3 * 0.2 * 10)

    # ========
    # Multiple trap species
    # ========
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.7],
            [0.2, 0.4, 0.3],
            [0.1, 0.3, 0.2],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    A1_trap_species = [test_species_1, test_species_2]
    height_e = 0.6

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert e_capt == pytest.approx((0.1 + 0.06) * 10 + (0.15 + 0.07) * 8)

    return 0


def test_update_trap_wmk_capture():
    """ Test update_trap_wmk_capture(). """

    # ========
    # First capture
    # ========
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    height_e = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # New highest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.9

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.9, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # Middle new watermark (a)
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.6

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.6, 1], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # Middle new watermark (b)
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.75

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.75, 1], [0.05, 0.3], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # New lowest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.3

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [[0.3, 1], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]],
        )
    )

    # ========
    # Multiple trap species
    # ========
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.7, 0.6],
            [0.2, 0.4, 0.3, 0.2],
            [0.1, 0.3, 0.2, 0.1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    height_e = 0.6

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [
                [0.6, 1, 1, 1],
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.3, 0.2, 0.1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
    )

    return 0


def test_update_trap_wmk_capture_not_enough():
    """ Test update_trap_wmk_capture_not_enough(). """

    # ========
    # First capture
    # ========
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    height_e = 0.5
    enough = 0.7

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.5, 0.7], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # New highest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.8
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.5, 0.9], [0.2, 0.7], [0.1, 0.5], [0, 0], [0, 0], [0, 0],],)
    )

    # ========
    # Middle new watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.6
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [[0.5, 0.9], [0.1, 0.7], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0],],
        )
    )

    # ========
    # New lowest watermark
    # ========
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.3
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [[0.3, 0.9], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]],
        )
    )

    # ========
    # Multiple trap species
    # ========
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.7, 0.6],
            [0.2, 0.4, 0.3, 0.2],
            [0.1, 0.3, 0.2, 0.1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    height_e = 0.6
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [
                [0.5, 0.9, 0.85, 0.8],
                [0.1, 0.7, 0.65, 0.6],
                [0.1, 0.4, 0.3, 0.2],
                [0.1, 0.3, 0.2, 0.1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
    )

    return 0


def test_capture_electrons_in_pixel():
    """ Test capture_electrons_in_pixel(). """

    # ========
    # First capture
    # ========
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    e_avai = 2500  # --> height_e = 0.5

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert e_capt == pytest.approx(5)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],)
    )

    # ========
    # Multiple trap species
    # ========
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.7],
            [0.2, 0.4, 0.3],
            [0.1, 0.3, 0.2],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    A1_trap_species = [test_species_1, test_species_2]
    e_avai = 3600  # --> height_e = 0.6

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert e_capt == pytest.approx((0.1 + 0.06) * 10 + (0.15 + 0.07) * 8)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [
                [0.6, 1, 1],
                [0.1, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
    )

    # ========
    # Not enough, first capture
    # ========
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    e_avai = 2.5e-3  # --> height_e = 4.9999e-4, enough = 0.50001

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert e_capt == pytest.approx(0.0025)
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array([[4.9999e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    )

    # ========
    # Not enough, multiple trap species
    # ========
    A2_trap_wmk_height_fill = np.array(
        [
            [0.5, 0.8, 0.7],
            [0.2, 0.4, 0.3],
            [0.1, 0.3, 0.2],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    A1_trap_species = [test_species_1, test_species_2]
    e_avai = 4.839e-4  # --> height_e = 2.199545e-4, enough = 0.5

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert e_capt == pytest.approx(2.199545e-4 * (0.1 * 10 + 0.15 * 8))
    assert A2_trap_wmk_height_fill == pytest.approx(
        np.array(
            [
                [2.199545e-4, 0.9, 0.85],
                [0.5 - 2.199545e-4, 0.8, 0.7],
                [0.2, 0.4, 0.3],
                [0.1, 0.3, 0.2],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
    )

    return 0


# //////////////////////////////////////////////////////////////////////////// #
#                               Main                                           #
# //////////////////////////////////////////////////////////////////////////// #
if __name__ == "__main__":
    test_create_express_multiplier()

    test_release_electrons_in_pixel()

    test_capture_electrons()
    test_update_trap_wmk_capture()
    test_update_trap_wmk_capture_not_enough()
    test_capture_electrons_in_pixel()
