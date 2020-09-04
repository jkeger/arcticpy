""" 
Correct CTI in an image from the Hubble Space Telescope (HST) Advanced Camera 
for Surveys (ACS) instrument.

It takes a while to correct a full image, so a slice of columns is extracted 
for this example.
"""

import arcticpy as ac
import os
from autoconf import conf
import matplotlib.pyplot as plt

# Path to this file
path = os.path.dirname(os.path.realpath(__file__))

# Set up some configuration options for the automatic fits dataset loading
conf.instance = conf.Config(config_path=f"{path}/config")

# Load the HST ACS dataset
frame = ac.acs.FrameACS.from_fits(
    file_path=f"{path}/acs/j8xi61meq_raw.fits", quadrant_letter="A"
)

# Extract a thin slice of a few columns as an example
frame = frame[:, 1000:1010]

# Plot the initial image
plt.figure()
im = plt.imshow(X=frame, aspect="equal", vmax=2800)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/acs/j8xi61meq_raw.png", dpi=400)
plt.close()

# Set CCD, ROE, and trap parameters for HST ACS at this Julian date
traps, ccd, roe = ac.model_for_HST_ACS(
    date=2400000.5 + frame.exposure_info.modified_julian_date
)

# Remove CTI
image_cti_removed = ac.remove_cti(
    image=frame,
    iterations=3,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
    parallel_express=2,
)

# Plot the corrected image
plt.figure()
im = plt.imshow(X=image_cti_removed, aspect="equal", vmax=2800)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/acs/j8xi61meq_corrected.png", dpi=400)
plt.close()

# Plot the difference
plt.figure()
im = plt.imshow(X=image_cti_removed - frame, aspect="equal")
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/acs/j8xi61meq_diff.png", dpi=400)
plt.close()

# Save the corrected image
frame_corrected.output_to_fits(
    file_path=f"{path}/acs/j8xi61meq_corrected.fits", overwrite=True
)
