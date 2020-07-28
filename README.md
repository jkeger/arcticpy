ArCTIC python
=============

AlgoRithm for Charge Transfer Inefficiency (CTI) Correction
-----------------------------------------------------------

Add or remove image trails due to charge transfer inefficiency in CCD detectors
by modelling the trapping, releasing, and moving of charge along pixels.

https://github.com/jkeger/arcticpy

Jacob Kegerreis: jacob.kegerreis@durham.ac.uk  
Richard Massey: r.j.massey@durham.ac.uk  
James Nightingale  

This file contains general documentation and some examples. See also the 
docstrings and comments throughout the code for further details, and the unit 
tests for examples and tests.


Contents
--------
+ Minimal usage
+ Files
+ Installation
+ Documentation
    + Add/remove CTI
    + CCD
    + ROE
    + Trap species
    + Trap managers
+ Examples 
    + Correcting HST images


Minimal usage
-------------
See the documentation below and the code docstrings for more detail.
```python
import arcticpy as ac

image_pre_cti = ... # Initial image

# CCD, ROE, and trap species parameters
ccd = ac.CCD(well_fill_power=0.8, full_well_depth=1e5)
roe = ac.ROE()
traps = [ac.Trap(density=3.141, release_timescale=1.234)]

image_post_cti = ac.add_cti(
    image=image_pre_cti,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
)

image_undo_cti = ac.remove_cti(
    image=image_post_cti,
    iterations=5,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
)
```



Files
-----
A summary of the code files and their content for reference. 

See the documentation below and the code docstrings for more details.

+ `arcticpy/` Main code directory
    + `main.py` 
        Contains the primary user-facing functions `add_cti()` and `remove_cti()`.
        
        Both of these are wrappers for `_clock_charge_in_one_direction()`,
        which contains the primary nested for loops over an image to add CTI to 
        (in order) each column, express (see below), and row.
    + `ccd.py` 
        Defines the user-facing `CCD` class that describes how electrons fill 
        the volume inside each (phase of a) pixel in a CCD detector.
    + `roe.py`  
        Defines the user-facing `ROE` class that describes the properties of 
        readout electronics (ROE) used to operate a CCD detector.
    + `traps.py`  
        Defines the user-facing `Trap` class that describes the properties of a 
        single species of traps.    
    + `trap_managers.py`  
        Defines the internal `TrapManager` class that organises one or many 
        species of traps.
        
        Contains the core function `n_electrons_released_and_captured()` called
        by `_clock_charge_in_one_direction()` to model the capture and release
        of electrons and to track the trapped electrons using the "watermarks".
    + `util.py`  
        Miscellaneous internal utilities.
+ `test_arcticpy/` Unit test directory
    + `test_main.py` General add/remove CTI tests
    + `test_ccd.py` CCD tests
    + `test_roe.py` ROE tests
    + `test_traps.py` Traps and trap manager tests



Installation
------------
+ Install the package with `pip install arcticpy`, see https://pypi.org/project/arcticpy/
+ Or download the code with `git clone https://github.com/jkeger/arcticpy.git`
    + Run the unit tests with `pytest test_arcticpy/`
+ Requires:
    + Python 3 (tested with 3.6.9)
    + ...



Documentation
=============
The code docstrings contain the full documentation. This section provides an 
overview of the basic functions and different options available for reference.

It is aimed at general users with a few extra details for anyone wanting to
navigate or work on the code itself.


Add/remove CTI
--------------
### Add CTI
To add (and remove) CTI trails, the primary inputs are the initial image and the 
properties of the CCD, readout electronics (ROE), and trap species, for either 
or both parallel and serial clocking:
```python
image_with_cti = ac.add_cti(
    image,
    parallel_ccd=None,
    parallel_roe=None,
    parallel_traps=None,
    parallel_express=0,
    parallel_offset=0,
    parallel_window=None,
    serial_ccd=None,
    serial_roe=None,
    serial_traps=None,
    serial_express=0,
    serial_offset=0,
    serial_window=None,
    time_window=[0, 1],
)
```
These are input using the `CCD`, `ROE`, and `Trap` classes, which are described 
in the sections below.


### Image
The input image should be a 2D array of (float) charge values, where the first 
dimension runs over the separate columns and the second inner dimension runs 
over the rows of pixels within each column.

By default, charge is clocked towards element 0, such that the charge in the 
first element/pixel 0 undergoes 1 transfer, and the final row N is furthest from 
the readout register so undergoes N+1 transfers.

Parallel clocking is the transfer along each independent column, while serial 
clocking is across the columns and is performed after parallel clocking, if the 
parameters for both are not `None`.

Note that instead of actually moving the charges past the traps in each pixel,
as happens in the real hardware, the code tracks the occupancies of the traps 
(see Watermarks below) and updates them by scanning over each pixel. This 
simplifies the code structure and keeps the image array conveniently static.


### Express
As described in Massey et al. (2014) section 2.1.5 in more detail, the effect of 
each individual pixel-to-pixel transfer can be similar, so multiple transfers 
can be computed efficiently at once.

This allows faster computation with a mild decrease in accuracy. 

For example, the electrons in the pixel closest to the readout have only one 
transfer, those in the 2nd pixel undergo 2, those in the 3rd have 3, and so on.

The `express` input sets the number of times the transfers are calculated. 
`express = 1` is the fastest and least accurate, `express = 2` means the 
transfers are re-computed half-way through the readout, up to `express = N` 
where `N` is the total number of pixels for the full computation of every step
without assumptions.

The default `express = 0` is a convenient alternative input for `express = N`.


### Offsets and windows
...


### Remove CTI
Removing CTI trails is done by iteratively modelling the addition of CTI, as 
described in Massey et al. (2010) section 3.2 and Table 1.

The `remove_cti()` function takes all the same parameters as `add_cti()`
plus the number of iterations for the forward modelling:
```python
image_remove_cti = ac.remove_cti(
    image=image_with_cti,
    iterations=5,
    # Same as add_cti()
)
```
More iterations provide higher accuracy at the cost of longer runtime. 
In practice, just 1 to 3 iterations are usually sufficient.



CCD
---
How electrons fill the volume inside each (phase of a) pixel.

See the `CCD` class docstring in `ccd.py` for the full documentation.

By default, charge is assumed to move instantly from one pixel to another, but 
multiple phases per pixel can be used in the CCD and ROE classes for more 
complicated multi-phase clocking or modelling of trap pumping etc.

### Default CCD
```python
ccd = ac.CCD(
    # The maximum number of electrons that can be contained within a pixel
    full_well_depth=1e4, 
    # The number of electrons that fit inside a 'notch' at the bottom of a 
    # potential well, avoiding any traps
    well_notch_depth=0.0,
    # The exponent in the power-law model for the volume occupied by the electrons
    well_fill_power=0.58,
    # For multiphase clocking: the proportion of traps within each phase or, in 
    # other words, the relative physical width (in any units) of each phase
    fraction_of_traps_per_phase=[1], 
    # Equivalent to the notch but for surface traps, defaults to full_well_depth
    well_bloom_level=None,
)
```

### Multi-phase clocking
```python
# Four phases of different widths, each with different well depths but the same 
# other values of the fill power and the default notch depth
ccd = ac.CCD(
    fraction_of_traps_per_phase=[0.5, 0.2, 0.2, 0.1],
    full_well_depth=[8.47e4, 1e5, 2e5, 3e5],
    well_fill_power=0.8,
)
```



ROE
---
The properties of readout electronics (ROE) used to operate a CCD.

See the `ROE` and child class docstrings in `roe.py` for the full documentation.

### Default ROE  
```python
roe = ac.roe(
    # The time between steps in the clocking sequence, in the same units as 
    # the trap capture and release timescales
    dwell_times=[1],
    # True: explicitly models the first pixel-to-pixel transfer differently to 
    # the rest by starting with empty traps for every pixel
    empty_traps_at_start=True,
    # True: columns have independent traps. False: columns move through the same
    # traps, may be appropriate for serial clocking
    empty_traps_between_columns=True,
    # ...
    force_downstream_release=True,
    # The type for the express multipliers, can be int for backwards compatibility
    express_matrix_dtype=float,
)
```
For multi-phase clocking, `dwell_times` should be the same length as the CCD's
`fraction_of_traps_per_phase`.


### Charge injection  
...


### Trap pumping  
...


### Express matrix  
The `ROE` class also contains the `express_matrix_from_pixels_and_express()` 
function used to generate the express matrix.



Trap species
------------
The parameters for a trap species.

See the `Trap` and child class docstrings in `traps.py` for the full documentation.

### Default trap species
```python
trap = ac.Trap(
    # The density of the trap species in a pixel
    density=0.13, 
    # The release timescale of the trap 
    release_timescale=0.25, 
    # The capture timescale of the trap, default 0 for instant capture
    capture_timescale=0, 
    # ...
    surface=False,
)
```
Combined release and capture, allowing for instant or non-zero capture times,
following Lindegren (1998) section 3.2.


### Instant capture
For the old algorithm of release first then instant-capture, used in previous 
versions of ArCTIC.
```python
trap = ac.TrapInstantCapture(density=0.13, release_timescale=0.25)
```


### Continuum distributions of lifetimes
For a trap species with a log-normal distribution of release lifetimes.
```python
trap = ac.TrapLogNormalLifetimeContinuum(
    # The total density of the trap species in a pixel
    density=10,
    # The median timescale of the distribution
    release_timescale_mu=1,
    # The sigma timescale for the width of the distribution
    release_timescale_sigma=0.5,
)
```
This is a special case of the `TrapLifetimeContinuum` class which allows 
any arbitrary distribution to be used instead of log-normal.

Uses the same combined release and capture algorithm as the default traps.



Trap managers
-------------
This is not relevant for typical users, but is key to the internal structure of 
the code and the "watermark" approach to tracking the trap occupancies.

The different trap managers also implement the different algorithms required for
the corresponding types of trap species described above, primarily with their 
`n_electrons_released_and_captured()` method.

See the `TrapManager` and child class docstrings in `trap_managers.py` for the 
full documentation.

### Watermarks 
A 2D array of fractional volumes -- the watermark levels -- and the 
corresponding fill fractions of each trap species:  
`[[volume, fill, fill, ...], [volume, fill, fill, ...], ... ]`  
This way, the proportion of filled traps can be tracked efficiently through the 
release and capture processes, including for multiple trap species.

In the simplest sense, capture of electrons creates a new watermark level at the 
"height" of the charge cloud and can overwrite lower ones as the traps are 
filled up, while release lowers the fraction of filled traps in each watermark 
level. Note that the stored volumes give the size of that level, not the 
cumulative total.

The details can vary for different trap managers, e.g. `TrapManagerTrackTime`,
which stores the time elapsed since the traps were filled instead of the fill 
fraction, in order to manage trap species with a continuum distribution of 
lifetimes.

All the trap species managed by one trap manager must use watermarks in the same 
way. If different trap types are used, then multiple trap managers are created
(see the `AllTrapManager` class).



Examples
========

Correcting HST images
---------------------


