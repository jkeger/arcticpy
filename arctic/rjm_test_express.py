import numpy as np
import pytest
from scipy import integrate
import matplotlib.pyplot as plt
from copy import deepcopy

import arctic as ac


# Plotting test -- manually set True to make the plot
# do_plot = False
do_plot = True

ccd_volume = ac.CCDVolume(
    well_fill_beta=0.5, well_max_height=2e5, well_notch_depth=1e-5
)
parallel_clocker = ac.Clocker(charge_injection=False)

density = 100
lifetime = 3
parallel_express = 0
parallel_offset = 0
size = 50
background = 1000
roi = (4, 9)
roi = None
cil = 1e5
pixels = np.arange(size)
image_orig = np.zeros((size, 1)) + background
image_orig[3:6, 0] = cil
image_orig[10:16, 0] = cil
# image_orig[20:26, 0] = cil
# image_orig[30:36, 0] = cil
# image_orig[40:46, 0] = cil
image_orig[35, 0] = 1e7  # 3.5*ccd_volume.well_max_height

plt.figure()

# Single trap
trap = ac.Trap(density=density, lifetime=lifetime)
# trap = ac.TrapSlowCapture(density=density, lifetime=lifetime, capture_timescale=2.0)
image_cti = ac.add_cti(
    image=image_orig,
    parallel_traps=[trap],
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
plt.show(block=False)

if do_plot:
    # Different express options
    for express in [1, 2, 3, 7, 11]:
        image_cti = ac.add_cti(
            image=image_orig,
            parallel_express=express,
            parallel_traps=[trap],
            parallel_ccd_volume=ccd_volume,
            parallel_clocker=parallel_clocker,
            parallel_offset=parallel_offset,
            parallel_roi=roi,
        )
        print((image_cti)[0:9, 0])
        plt.plot(
            pixels, (image_cti - background)[:, 0], label=r"Express$=%.1d$" % express
        )

plt.legend()
plt.yscale("log")
plt.xlabel("Pixel")
plt.ylabel("Counts")


plt.show(block=False)
input("Press Enter to continue...")
