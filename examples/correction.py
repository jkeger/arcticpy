"""
__Example: Advanced Camera for Surveys (ACS) CTI Correction__

This example uses **arcticpy** to correct CTI in an example image from the 
Hubble Space Telescope (HST) Advanced Camera for Surveys (ACS) instrument.
"""

""" Set up the path to this example script. """

import os

path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

"""
We use this path to set:
    config_path:
        Where arcticpy configuration files are located. They control many 
        aspects of arcticpy (visualization, model priors, etc.).
"""

from autoconf import conf

conf.instance = conf.Config(config_path=f"{path}/config")

"""
Load the ACS example dataset 'j8xi61meq_raw.fits' from .fits files.
"""

import arcticpy as ac

dataset_label = "acs"
dataset_name = "j8xi61meq_raw"
dataset_path = f"{path}/{dataset_label}/{dataset_name}"

"""
Imaging data observed on the ACS consists of four quadrants ('A', 'B', 'C', 'D') 
which have the following layout:

       <--------S------------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []    /\
    /  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] /     | Direction arctic
    |  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |     | clocks an image
    P  [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx] P     | without rotation
    |  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |     | (i.e. towards row 
    \/ [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] \/    | 0 of the NumPy 
                                                             | arrays)

For a ACS .fits file:

 - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
 - The images contained in hdu 4 correspond to quadrants C (left) and D (right).

The _FrameACS_ contains all the functionality we need to automatically load ACS 
data from the above .fits structure, without requiring us to manually set up the 
dataset for CTI correction. This includes:

 - Loading the input quadrant from the .fits file, from the correct .fits HDU 
    extension.
 - Extracting the quadrant from this array.
 - Rotating this extracted quadrant such that for our CTI correction 
    **arcticpy** clocks the image is the appropriate direction.
 - Loading and storing exposure information (e.g. date of observation, exposure 
    time).
 - Using this information to convert the units of the image to counts, which are 
    the appropriate unit for **arcticpy**.

"""

frame = ac.acs.FrameACS.from_fits(file_path=f"{dataset_path}.fits", quadrant_letter="A")

"""
We can use the **arcticpy** visualization libary to plot this _Frame_. 

The image wouldn't not display clearly using the default color scale (which is 
the default color scale of matplotlib) so we've updated it below.
"""

plotter = ac.plot.Plotter(cmap=ac.plot.ColorMap(norm="linear", norm_max=2800.0))
ac.plot.Frame(frame=frame, plotter=plotter)

"""
To correct the ACS data, we need a CTI model describing:
 
 - The properties of the _Trap_'s on the CCD that are responsible for CTI by 
    capturing and releasing electrons during read-out.
 - The volume-filling behaviour of electrons in CCD pixels, as this effects 
    whether _Trap_'s are able to captrue them or not!
   
**arcticpy** again has in-built tools for automatically loading the CTI model 
appropriate for correcting ACS data, using the date of observation loaded in the 
_ACSFrame_ object to choose a model based on the date of observation.
"""

traps, ccd, roe = ac.model_for_HST_ACS(date=2453347.925)

"""
We can now pass the image and CTI model to remove CTI, which mimics the effects 
of CTI and in turn use this clocked image to perform the CTI correction.

For simplicity, we'll use all default values except using 5 iterations, which 
means when correcting the image we clock it to reproduce CTI and use this image 
to correct the image, and repeat that process 5 times.
"""

frame_corrected = ac.remove_cti(
    image=frame,
    iterations=1,
    parallel_express=1,
    parallel_traps=traps,
    parallel_ccd=ccd,
)
ac.plot.Frame(frame=frame_corrected, plotter=plotter)

"""
Finally, we output this corrected image to .fits, so we can use it in the 
future.

[Currently, we output the image to a .fits file called 'j8xi61meq_corrected', 
which does NOT use the same 6 slice multi extension fits format of the original 
data. We are adding a method commented out below called 'update_fits', which 
will allow the output to be in a file of this format. However this is currently 
bugged and disabled!]
"""

frame_corrected.output_to_fits(
    file_path=f"{path}/{dataset_label}/j8xi61meq_corrected", overwrite=False
)
