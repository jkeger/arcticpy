import numpy as np
import pytest
from scipy import integrate
import matplotlib.pyplot as plt
from copy import deepcopy

import arctic as ac


np.set_printoptions(threshold=160, linewidth=160)

# Plotting test -- manually set True to make the plot
# do_plot = False
do_plot = True

ccd_volume = ac.CCDVolume(well_fill_beta=0.6, well_max_height=2e5, well_notch_depth=1)
parallel_clocker = ac.Clocker(charge_injection=True, empty_traps_at_start=True)

density = 10
lifetime = 3
parallel_express = 0
parallel_offset = 1000
size = 50
background = 0
roi = range(4, 29)
roi = None
cil = 1e5
pixels = np.arange(size)
image_orig = np.zeros((size, 1)) + background
image_orig[3:6, 0] = 2 * ccd_volume.well_max_height
# image_orig[10:16, 0] = cil
# image_orig[20:26, 0] = cil
# image_orig[40:46, 0] = cil
image_orig[40:46, 0] = cil

plt.figure()

# Single trap
trap = ac.Trap(density=density, lifetime=lifetime)
surface_trap_density = 1
blooming_level = ccd_volume.well_max_height * 0.99
blooming_height = ccd_volume.electron_fractional_height_from_electrons(blooming_level)
print(blooming_level, blooming_height)
# surface_trap = ac.TrapNonUniformHeightDistribution(
#    density=surface_trap_density,
#    lifetime=0.01,
#    electron_fractional_height_min=blooming_height,
#    electron_fractional_height_max=blooming_height,
# )
# trap = ac.TrapSlowCapture(density=density, lifetime=lifetime, capture_timescale=2.0)
# traps = [[trap], [surface_trap]]
traps = [trap]
image_cti = ac.add_cti(
    image=image_orig,
    parallel_express=0,
    parallel_traps=traps,
    parallel_ccd_volume=ccd_volume,
    parallel_clocker=parallel_clocker,
    parallel_offset=parallel_offset,
    parallel_roi=roi,
)
plt.scatter(
    pixels, (image_cti - background)[:, 0], c="k", marker=".", label="Express$=0$"
)


print("total of image ", np.sum(image_cti - background))

# Pure exponential for comparison
exp_trail = np.exp(-pixels[2:] / lifetime)
exp_trail *= (image_cti[size - 1, 0] - background) / exp_trail[size - 3]
plt.plot(pixels[2:], exp_trail, c="k", alpha=0.3)
plt.plot(pixels, image_orig, c="k", alpha=0.3)
plt.show(block=False)

if do_plot:
    # Different express options
    for express in [1, 2, 3, 5, 10]:  # , 28, 29, 50]:
        # for express in [2]:
        print()
        # for tf in [False, True]:
        # parallel_clocker.empty_traps_at_start = tf
        image_cti = ac.add_cti(
            image=image_orig,
            parallel_express=express,
            parallel_traps=traps,
            parallel_ccd_volume=ccd_volume,
            parallel_clocker=parallel_clocker,
            parallel_offset=parallel_offset,
            parallel_roi=roi,
        )
        plt.plot(
            pixels, (image_cti - background)[:, 0], label=r"Express$=%.1d$" % express
        )

        # Pure exponential for comparison
        exp_trail = np.exp(-pixels[2:] / lifetime)
        exp_trail *= (image_cti[size - 1, 0] - background) / exp_trail[size - 3]
        # plt.plot(pixels[2:], exp_trail, c="k", alpha=0.3)

plt.plot(pixels, image_orig, c="k", alpha=0.3)

plt.legend()
plt.yscale("log")
plt.xlabel("Pixel")
plt.ylabel("Counts")


plt.show(block=False)
input("Press Enter to continue...")
