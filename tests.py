""" CTI python tests

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk
    
    WIP...
"""
from cti import *


# //////////////////////////////////////////////////////////////////////////// #
#                               Test objects                                   #
# //////////////////////////////////////////////////////////////////////////// #
test_species_1 = TrapSpecies(density=10, lifetime=-1 / np.log(0.5))
test_species_2 = TrapSpecies(density=8, lifetime=-1 / np.log(0.2))


# //////////////////////////////////////////////////////////////////////////// #
#                               Test Functions                                 #
# //////////////////////////////////////////////////////////////////////////// #
def test_release_electrons_in_pixel():
    print("Test release_electrons_in_pixel()")

    # ========
    print("  Empty release ", end="")
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
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
    )
    print("[PASS]")

    # ========
    print("  Single trap species ", end="")
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
        np.array([[0.5, 0.4], [0.2, 0.2], [0.1, 0.1], [0, 0], [0, 0], [0, 0]]),
    )
    print("[PASS]")

    # ========
    print("  Multiple trap species ", end="")
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
        np.array(
            [
                [0.5, 0.4, 0.06],
                [0.2, 0.2, 0.04],
                [0.1, 0.1, 0.02],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    print("[PASS]")

    return 0


def test_capture_electrons():
    print("Test capture_electrons()")

    # ========
    print("  First capture ", end="")
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
    print("  New highest watermark ", end="")
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
    print("  Middle new watermark (a) ", end="")
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
    print("  Middle new watermark (b) ", end="")
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
    print("  New lowest watermark ", end="")
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
    print("  Multiple trap species ", end="")
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


# //////////////////////////////////////////////////////////////////////////// #
#                               Main                                           #
# //////////////////////////////////////////////////////////////////////////// #
if __name__ == "__main__":
    test_release_electrons_in_pixel()
    test_capture_electrons()
