""" Profile the code to test and debug/optimise the performance. 

Usage
-----
$  python3  test_arctic/profile_demo.py  express  output_name  do_plot

Args
----
express : int
    ArCTIc express parameter.
    
output_name : str (opt.)
    Saves the profiling output to `test_arctic/output_name.txt`, and the image 
    to `test_arctic/output_name.png`. Defaults to "test_<express>". 
    
do_plot : int (opt.)
    If 1 then plot the output, default 0.
"""

import os
import sys
import cProfile, pstats, io
import timeit
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import arctic as ac


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


def add_cti_to_hst_image(express=1):

    # Load the HST image
    Fp_image = "test_arctic/jc0a01h8q_raw.fits"
    hdu_list = fits.open(Fp_image)
    idx_image = 1
    input_image = np.array(hdu_list[idx_image].data).astype("float64").T

    # Model inputs
    trap = ac.Trap(density=1, release_timescale=10)
    ccd = ac.CCD(well_notch_depth=0.01, well_fill_power=0.8, full_well_depth=84700)

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


if __name__ == "__main__":
    # Input parameters
    try:
        express = int(sys.argv[1])
    except IndexError:
        express = 1
    print("express = %d" % express)
    try:
        output_name = sys.argv[2]
    except IndexError:
        output_name = "test_%d" % express
    print('output_name = "%s"' % output_name)
    try:
        do_plot = int(sys.argv[3])
    except IndexError:
        do_plot = 0
    print("do_plot = %d" % do_plot)

    # Time without profiling (manual toggle)
    if not True:

        def time_wrapper():
            return add_cti_to_hst_image(express=express)

        print("time: ", timeit.timeit(time_wrapper, number=1))

        # Write input image to new fits file e.g. for C++ comparison
        if not True:
            Fp_image = "test_arctic/input_image.fits"
            new_hdr = fits.Header()
            hdu = fits.PrimaryHDU(np.flipud(input_image), new_hdr)
            hdu.writeto(Fp_image)
            print("Saved input image %s" % Fp_image)

        exit()

    # Set up profiling
    pr = cProfile.Profile()
    pr.enable()

    # Add CTI
    input_image, output_image = add_cti_to_hst_image(express=express)
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
    if do_plot == 1:
        plot_counts(input_image, output_image, output_name=output_name)
        plot_difference(input_image, output_image, output_name=output_name)


# Notes
# -----
#
# express 1
#   5.5 s (no-profile 4.2 s) C++: 2 s
#
# express 2
#   6.7 s (no-profile 4.6 s) C++: 3 s
#
# express 5
#   11.2 s (no-profile 6.2 s) C++: 6 s
#   Diff columns: 13.3, 11.8, 11.5, 11.2, 11.6, 11.9, 11.6, 11.7
#
# express 10
#   17.1 s (no-profile 9.7 s) C++: 11 s
#
# express 20
#   37.5 s (no-profile 15.4 s) C++: 22 s
#
#
# OLD (bad express matrix sums)
# express 1
#   28 s (no-profile 21 s) C++: 2 s
#   Diff columns: 30, 28, 29, 27, 27, 30, 31 (next day: 23)
#
# express 2
#   30 s (no-profile 23 s) C++: 3 s
#
# express 5
#   33 s (no-profile 25 s) C++: 6 s
#
# express 10
#   41 s (no-profile 26 s) C++: 11 s
#
# express 20
#   51 s (no-profile 35 s) C++: 22 s
