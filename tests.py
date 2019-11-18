""" CTI python tests

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk
    
    WIP...
"""
from cti import *


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
    print("Test create_express_multiplier()")

    # ========
    print("    Full express ", end="")

    A2_express_multiplier = create_express_multiplier(express=1, num_row=12)

    assert np.allclose(A2_express_multiplier, np.arange(1, 13))
    print("[PASS]")

    # ========
    print("    Medium express ", end="")

    A2_express_multiplier = create_express_multiplier(express=4, num_row=12)

    assert np.allclose(
        A2_express_multiplier,
        [
            [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
        ],
    )
    print("[PASS]")

    # ========
    print("    No express ", end="")

    A2_express_multiplier = create_express_multiplier(express=12, num_row=12)

    assert np.allclose(A2_express_multiplier, np.triu(np.ones((12, 12))))
    print("[PASS]")


def test_release_electrons_in_pixel():
    print("Test release_electrons_in_pixel()")

    # ========
    print("    Empty release ", end="")
    A1_trap_species = [test_species_1]
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )

    e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
        A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_rele, 0)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Single trap species ", end="")
    A1_trap_species = [test_species_1]
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.2], [0, 0], [0, 0], [0, 0]]
    )

    e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
        A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_rele, 2.5)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Multiple trap species ", end="")
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

    assert np.isclose(e_rele, 2.5 + 1.28)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [
            [0.5, 0.4, 0.06],
            [0.2, 0.2, 0.04],
            [0.1, 0.1, 0.02],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )
    print("[PASS]")

    return 0


def test_capture_electrons():
    print("Test capture_electrons()")

    # ========
    print("    First capture ", end="")
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    height_e = 0.5

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_capt, 0.5 * 10)
    print("[PASS]")

    # ========
    print("    New highest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.9

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_capt, (0.1 + 0.12 + 0.07 + 0.1) * 10)
    print("[PASS]")

    # ========
    print("    Middle new watermark (a) ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.6

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_capt, (0.1 + 0.06) * 10)
    print("[PASS]")

    # ========
    print("    Middle new watermark (b) ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.75

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_capt, (0.1 + 0.12 + 0.035) * 10)
    print("[PASS]")

    # ========
    print("    New lowest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.3

    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    assert np.isclose(e_capt, 0.3 * 0.2 * 10)
    print("[PASS]")

    # ========
    print("    Multiple trap species ", end="")
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

    assert np.isclose(e_capt, (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8)
    print("[PASS]")

    return 0


def test_update_trap_wmk_capture():
    print("Test update_trap_wmk_capture()")

    # ========
    print("    First capture ", end="")
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    height_e = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    New highest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.9

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.9, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Middle new watermark (a) ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.6

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.6, 1], [0.1, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Middle new watermark (b) ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.75

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.75, 1], [0.05, 0.3], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    New lowest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.3

    A2_trap_wmk_height_fill = update_trap_wmk_capture(
        height_e, A2_trap_wmk_height_fill
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.3, 1], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Multiple trap species ", end="")
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

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [
            [0.6, 1, 1, 1],
            [0.1, 0.4, 0.3, 0.2],
            [0.1, 0.3, 0.2, 0.1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )
    print("[PASS]")

    return 0


def test_update_trap_wmk_capture_not_enough():
    print("Test update_trap_wmk_capture_not_enough()")

    # ========
    print("    First capture ", end="")
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    height_e = 0.5
    enough = 0.7

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.5, 0.7], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    New highest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.8
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.25, 0.9], [0.25, 0.8], [0.1, 0.7], [0.1, 0.4], [0.1, 0.5], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Middle new watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    A1_trap_species = [test_species_1]
    height_e = 0.6
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.25, 0.9], [0.25, 0.8], [0.1, 0.7], [0.1, 0.4], [0.1, 0.3], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    New lowest watermark ", end="")
    A2_trap_wmk_height_fill = np.array(
        [[0.5, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0], [0, 0]]
    )
    height_e = 0.3
    enough = 0.5

    A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
        height_e, A2_trap_wmk_height_fill, enough
    )

    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.3, 0.9], [0.2, 0.8], [0.2, 0.4], [0.1, 0.3], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Multiple trap species ", end="")
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
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [
            [0.25, 0.9, 0.85, 0.8],
            [0.25, 0.8, 0.7, 0.6],
            [0.1, 0.7, 0.65, 0.6],
            [0.1, 0.4, 0.3, 0.2],
            [0.1, 0.3, 0.2, 0.1],
            [0, 0, 0, 0],
        ],
    )
    print("[PASS]")

    return 0


def test_capture_electrons_in_pixel():
    print("Test capture_electrons_in_pixel()")

    # ========
    print("    First capture ", end="")
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    e_avai = 2500  # --> height_e = 0.5

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert np.isclose(e_capt, 5)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[0.5, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Multiple trap species ", end="")
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

    assert np.isclose(e_capt, (0.1 + 0.06) * 10 + (0.15 + 0.07) * 8)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [
            [0.6, 1, 1],
            [0.1, 0.4, 0.3],
            [0.1, 0.3, 0.2],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )
    print("[PASS]")

    # ========
    print("    Not enough, first capture ", end="")
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(
        num_column=6, num_species=1
    )
    A1_trap_species = [test_species_1]
    e_avai = 2.5e-3  # --> height_e = 5e-4, enough = 0.50001

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )

    assert np.isclose(e_capt, 0.0025)
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [[5e-4, 0.50001], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    )
    print("[PASS]")

    # ========
    print("    Not enough, Multiple trap species ", end="")
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
    e_avai = 4.839e-4  # --> height_e = 2.1995e-4, enough = 0.5

    e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
        e_avai, A2_trap_wmk_height_fill, A1_trap_species, test_ccd_1
    )
    assert np.isclose(e_capt, 2.1995e-4 * (0.1 * 10 + 0.15 * 8))
    assert np.allclose(
        A2_trap_wmk_height_fill,
        [
            [2.1995e-4, 0.9, 0.85],
            [0.5 - 2.1995e-4, 0.8, 0.7],
            [0.2, 0.4, 0.3],
            [0.1, 0.3, 0.2],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )
    print("[PASS]")

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
