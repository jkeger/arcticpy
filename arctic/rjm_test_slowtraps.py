import numpy as np
import pytest
from scipy import integrate
import matplotlib.pyplot as plt
from copy import deepcopy

import arctic as ac

# Define trap parameters
density = 100
lifetime = 3
capture_timescale = 2.0
trap = ac.Trap(density=density, lifetime=lifetime)


# Define input image
size = 50
background = 0
offset = 0
cil = 1e5
image_orig = np.zeros((size, 1)) + background
image_orig[3:6, 0] = cil
image_orig[10:16, 0] = cil
image_orig[20:26, 0] = cil
image_orig[30:36, 0] = cil
image_orig[40:46, 0] = cil
roi = None  # range(11, 25)

# Define instrument parameters
ccd_volume = ac.CCDVolume(well_fill_beta=1, well_max_height=2e5, well_notch_depth=0)
parallel_clocker = ac.Clocker(charge_injection=False)
express = 1


# Add CTI
image_cti = ac.add_cti(
    image=image_orig,
    parallel_express=express,
    parallel_offset=offset,
    parallel_roi=roi,
    parallel_traps=[trap],
    parallel_ccd_volume=ccd_volume,
    parallel_clocker=parallel_clocker,
)


plt.figure()
pixels = np.arange(size)
plt.scatter(
    pixels,
    (2 + image_cti - background)[:, 0],
    c="k",
    marker=".",
    label="Instant capture",
)
# print("total of image ",np.sum(image_cti-background))

# Pure exponential for comparison
exp_trail = np.exp(-pixels[2:] / lifetime)
exp_trail *= image_cti[size - 1, 0] / exp_trail[size - 3]
plt.plot(pixels[2:], 2 + exp_trail - background, c="k", alpha=0.3)
plt.plot(pixels, image_orig - background, c="k", alpha=0.3, label="input image")
plt.show(block=False)

# Different express options
for tau_c in [0.00001, 1, 10]:
    trap = ac.TrapSlowCapture(
        density=density, lifetime=lifetime, capture_timescale=tau_c
    )

    image_cti = ac.add_cti(
        image=image_orig,
        parallel_express=express,
        parallel_offset=offset,
        parallel_roi=roi,
        parallel_traps=[trap],
        parallel_ccd_volume=ccd_volume,
        parallel_clocker=parallel_clocker,
    )
    # print((image_cti)[0:9, 0])
    plt.plot(pixels, (2 + image_cti - background)[:, 0], label=r"tau_c$=%.1f$" % tau_c)

plt.legend()
plt.yscale("log")
plt.xlabel("Pixel")
plt.ylabel("Counts")

# plt.show()
plt.show(block=False)
input("Press Enter to continue...")
