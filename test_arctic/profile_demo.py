""" Profile the code to test and debug/optimise the performance. 

Usage
-----
$  python3  test_arctic/profile_demo.py  output_name

Args
----
output_name : str
    Saves the profiling output to `test_arctic/output_name.txt`, and the image 
    to `test_arctic/output_name.png`. If not provided then defaults to "test".    
"""

import os
import sys
import cProfile, pstats, io
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import arctic as ac


def add_cti_to_hst_image():

    # Load the HST image
    Fp_image = "test_arctic/jc0a01h8q_raw.fits"
    # print("Loading %s ... " % Fp_image, end="")
    # sys.stdout.flush()
    hdu_list = fits.open(Fp_image)
    idx_image = 1
    input_image = np.array(hdu_list[idx_image].data).astype("float64").T
    # print("Done")

    # Model inputs
    trap = ac.Trap(density=1, release_timescale=10)
    ccd = ac.CCD(well_notch_depth=0.01, well_fill_power=0.8, full_well_depth=84700)
    express = 1

    # Select a subset of rows
    row_start = 0
    row_end = -1
    # row_start = 200
    # row_end = 400

    # Select a subset of columns
    column_start = 2662
    column_end = column_start + 1

    # Refine input image
    input_image = input_image[column_start:column_end, row_start:row_end].T
    print("%d row(s), %d column(s)" % input_image.shape)

    # Add CTI to the column
    output_image = ac.add_cti(
        image=input_image,
        parallel_traps=[trap],
        parallel_ccd=ccd,
        parallel_express=express,
    )

    return input_image, output_image


def plot_counts(input_image, output_image, output_name=None):
    """ Plot the counts of the output and input images. """
    pixels = np.arange(len(input_image))

    plt.figure(figsize=(12, 6))

    plt.plot(pixels, input_image, lw=0.2, alpha=0.8, label="Input")
    plt.plot(pixels, output_image, lw=0.2, alpha=0.8, label="CTI added")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("Pixel")
    plt.ylabel("Counts")
    plt.tight_layout()

    # Save or show the figure
    if output_name is not None:
        Fp_save = "test_arctic/%s_counts.png" % output_name
        plt.savefig(Fp_save, dpi=600)
        print("Saved %s" % Fp_save)
    else:
        plt.show()
    plt.close()


def plot_difference(input_image, output_image, output_name=None):
    """ Plot the difference between the output and input images. """
    pixels = np.arange(len(input_image))

    plt.figure(figsize=(12, 6))

    plt.plot(pixels, output_image - input_image, lw=0.4, alpha=0.8)

    plt.xlabel("Pixel")
    plt.ylabel("Count Difference")
    plt.tight_layout()

    # Save or show the figure
    if output_name is not None:
        Fp_save = "test_arctic/%s_diff.png" % output_name
        plt.savefig(Fp_save, dpi=600)
        print("Saved %s" % Fp_save)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        output_name = sys.argv[1]
    except IndexError:
        output_name = "test"

    # Set up profiling
    pr = cProfile.Profile()
    pr.enable()

    # Add CTI
    input_image, output_image = add_cti_to_hst_image()
    pr.disable()

    # Save profiling output
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    # Print first few lines and save all
    print(s.getvalue()[:1500], ". . .")
    Fp_output = "test_arctic/%s.txt" % output_name
    with open(Fp_output, "w") as f:
        f.write(s.getvalue())
        print("\nWrote %s" % Fp_output)

    # Plot image
    plot_counts(input_image, output_image, output_name=output_name)
    plot_difference(input_image, output_image, output_name=output_name)
