""" AlgoRithm for Charge Transfer Inefficiency Correction

    https://github.com/jkeger/arcticpy

    Jacob Kegerreis (2020) jacob.kegerreis@durham.ac.uk
    James Nightingale
    Richard Massey

    WIP...
"""
import numpy as np
import math
from copy import deepcopy

from arctic.clock import Clocker
from arctic.traps import (
    TrapNonUniformHeightDistribution,
    TrapLifetimeContinuum,
    TrapLogNormalLifetimeContinuum,
    TrapSlowCapture,
)
from arctic.trap_managers import (
    TrapManager,
    TrapManagerNonUniformHeightDistribution,
    TrapManagerTrackTime,
    TrapManagerSlowCapture,
)


def express_matrix_from_rows_and_express(
    rows, express=0, offset=0, roi=None, charge_injection=False
):
    """ 
    Calculate the values by which the effects of clocking each pixel must be 
    multiplied when using the express algorithm. See Massey et al. (2014), 
    section 2.1.5.

    Parameters
    ----------
    express : int
        The express parameter = (maximum number of transfers) / (number of computed tranfers), i.e. the number of 
        computed transfers per pixel. 
        So express = 1 --> maxmimum speed; 
           express = maximum number of transfers (or equivalently zero) --> maximum accuracy
    rows : int
        The number of rows in the image and hence the maximum number of transfers.

    Returns
    -------
    express_multiplier : [[float]]
        The express multiplier values for each pixel.
    """

    # Default to very slow but accurate behaviour
    express = rows if express == 0 else min(express, rows)

    # Initialise an array large enough to contain the supposed image, including offset
    # express_multiplier = np.empty((express, rows + offset), dtype=int)
    # There is no reason why the express multiplier need be an integer!
    express_multiplier = np.empty((express, rows + offset))

    # Calculate the maximum number of transfers that one actualised transfer can represent
    # express_max = math.ceil((rows + offset) / express)  # if it's going to be an integer, ceil() rather than int() is required in case the number of rows is not an integer multiple of express
    express_max = (rows + offset) / express

    if charge_injection:
        express_multiplier[:] = express_max
    else:
        express_multiplier[:] = np.arange(1, rows + offset + 1)
        # Offset each row to account for the pixels that have already been read out
        for express_index in range(express):
            express_multiplier[express_index] -= express_index * express_max
        # Set all values to between 0 and express_max
        express_multiplier[express_multiplier < 0] = 0
        express_multiplier[express_max < express_multiplier] = express_max
        #
        # RJM: could make a nice unit test that the following should always be [1,2,3,4,5]
        #
        # assert np.sum(express_multiplier,axis=0)

    # Extract the section of the array corresponding to the image (without the offset)
    express_multiplier = express_multiplier[:, offset + np.arange(rows)]
    express, rows = express_multiplier.shape

    # Set to zero every pixel outside the region of interest
    if roi is not None:
        inside_roi = np.zeros((express, rows), dtype=int)
        inside_roi[:, max(roi[0], 0) : min(roi[1] + 1, rows)] = 1
        express_multiplier *= inside_roi

    # Omit all rows containing only zeros, for speed later
    express_multiplier = express_multiplier[np.sum(express_multiplier, axis=1) > 0, :]
    n_express = (express_multiplier.shape)[0]

    # Remember appropriate moments to store trap occupancy levels, so next EXPRESS iteration
    # can continue from an (approximately) suitable configuration
    when_to_store_traps = np.zeros((n_express, rows), dtype=bool)
    for express_index in range(n_express - 1):
        for row_index in range(rows):
            if express_multiplier[express_index + 1, row_index] > 0:
                break
        when_to_store_traps[express_index, row_index] = True

    return express_multiplier, when_to_store_traps


def _add_cti_to_image(
    image, clocker, ccd_volume, traps, express, offset, roi_rows, roi_columns
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values.
    clocker : Clocker
        The object describing the clocking of electrons with read-out 
        electronics.
    ccd_volume : CCDVolume
        The object describing the CCD volume. 
    traps : [Trap] or [[Trap]]
        A list of one or more trap objects. To use different types of traps 
        that will require different watermark levels, pass a 2D list of 
        lists, i.e. a list containing lists of one or more traps for each 
        type. 
    express : int
        The factor by which pixel-to-pixel transfers are combined for 
        efficiency.

    Returns
    -------
    image : np.ndarray
        The output array of pixel values.
    """

    rows, columns = image.shape

    # Make sure the arrays are arrays
    if not isinstance(traps[0], list):
        traps = [traps]

    phases = len(clocker.sequence)
    assert len(ccd_volume.phase_widths) == phases
    assert np.amax(ccd_volume.phase_widths) <= 1

    # Default to no express and calculate every step
    # express = rows if express == 0 else min(express,rows)
    # print("help",rows,columns,express)

    express_matrix, when_to_store_traps = express_matrix_from_rows_and_express(
        rows=rows,
        express=express,
        offset=offset,
        roi=roi_rows,
        charge_injection=clocker.charge_injection,
    )
    n_express = (express_matrix.shape)[0]
    # print(express_matrix)

    # Prepare the image and express for multi-phase clocking
    if phases > 1:
        new_image = np.zeros((rows * phases, columns))
        new_image[ccd_volume.integration_phase :: phases] = image
        image = new_image
        rows, columns = image.shape

        express_matrix = np.repeat(express_matrix, phases, axis=1)

    # Set up the array of trap managers
    trap_managers = []
    for trap_group in traps:
        # Use a non-default trap manager if required for the input trap species
        if isinstance(trap_group[0], TrapNonUniformHeightDistribution):
            trap_managers.append(
                TrapManagerNonUniformHeightDistribution(traps=trap_group, rows=rows)
            )
        elif isinstance(
            trap_group[0], (TrapLifetimeContinuum, TrapLogNormalLifetimeContinuum),
        ):
            trap_managers.append(TrapManagerTrackTime(traps=trap_group, rows=rows))
        elif isinstance(trap_group[0], TrapSlowCapture):
            trap_managers.append(TrapManagerSlowCapture(traps=trap_group, rows=rows))
        else:
            trap_managers.append(TrapManager(traps=trap_group, rows=rows))

    # Run each independent column of pixels
    for column_index in range(columns):

        # Each calculation of the effects of traps on the pixels
        for express_index in range(n_express):

            # print(express_matrix[express_index, :])

            # Reset the trap states
            if express_index == 0:
                for trap_manager in trap_managers:
                    trap_manager.reset_traps_for_next_express_loop()
            else:
                trap_managers = stored_trap_managers
                print("restoring trap occupancy")

            # Each pixel
            for row_index in range(rows):
                #    print(express_index, row_index)
                express_multiplier = express_matrix[express_index, row_index]
                if express_multiplier == 0:
                    continue

                # Save trap occupancy
                if when_to_store_traps[express_index, row_index]:
                    stored_trap_managers = trap_managers

                # Initial number of electrons available for trapping
                electrons_initial = image[row_index, column_index]

                # Release and capture
                phase = row_index % phases
                total_electrons_released_and_captured = 0
                for trap_manager in trap_managers:
                    net_electrons_released_and_captured = trap_manager.electrons_released_and_captured_in_pixel(
                        electrons_available=electrons_initial,
                        dwell_time=clocker.sequence[phase],
                        ccd_volume=ccd_volume.extract_phase(phase),
                        width=ccd_volume.phase_widths[phase],
                    )

                    total_electrons_released_and_captured += (
                        net_electrons_released_and_captured
                    )
                    # print("hello")
                    # print(trap_manager.watermarks[:,0])
                    # print(trap_manager.watermarks[:,1])
                    # input("Press Enter to continue...")

                # Total change to electrons in pixel
                electrons_initial += (
                    total_electrons_released_and_captured * express_multiplier
                )

                # if row_index < 12: print(row_index,express_multiplier,total_electrons_released_and_captured)

                image[row_index, column_index] = electrons_initial

    # Recombine the image for multi-phase clocking
    if phases > 1:
        image = image.reshape((int(rows / phases), phases, columns)).sum(axis=1)

    return image


def add_cti(
    image,
    parallel_express=0,
    parallel_charge_injection_mode=False,
    parallel_clocker=None,
    parallel_ccd_volume=None,
    parallel_traps=None,
    parallel_offset=0,
    parallel_roi=None,
    serial_express=0,
    serial_clocker=None,
    serial_ccd_volume=None,
    serial_traps=None,
    serial_offset=0,
    serial_roi=None,
    prefill_traps=True,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns, for parallel and/or serial clocking.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values.
    parallel_express : int
        The factor by which pixel-to-pixel transfers are combined for 
        efficiency for parallel clocking.
    parallel_charge_injection_mode : bool
        If True, parallel clocking is performed in charge injection line 
        mode, where each pixel is clocked and therefore trailed by traps 
        over the entire CCD (as opposed to its distance from the CCD register).
    parallel_clocker : Clocker
        The object describing the clocking read-out electronics for parallel 
        clocking.
    parallel_ccd_volume : CCDVolume
        The object describing the CCD volume for parallel clocking. For 
        multi-phase clocking optionally use a list of different CCD volumes
        for each phase, in the same size list as parallel_clocker.sequence.
    parallel_traps : [Trap] or [[Trap]]
        A list of one or more trap objects for parallel clocking. To use 
        different types of traps that will require different watermark 
        levels, pass a 2D list of lists, i.e. a list containing lists of 
        one or more traps for each type.
    parallel_offset : int
        The supplied image array is a postage stamp offset this number of 
        pixels from the readout register
    serial_* : *
        The same as the parallel_* objects described above but for serial 
        clocking instead.
    region_of_interest : (

    Returns
    -------
    image : np.ndarray
        The output array of pixel values.
    """

    # If Clocker not provided then assume simple, single-phase clocking
    if parallel_clocker is None:
        parallel_clocker = Clocker()
    if serial_clocker is None:
        serial_clocker = Clocker()

    # Don't modify the external array passed to this function
    image = deepcopy(image)

    if parallel_traps is not None:

        image = _add_cti_to_image(
            image=image,
            clocker=parallel_clocker,
            traps=parallel_traps,
            ccd_volume=parallel_ccd_volume,
            express=parallel_express,
            offset=parallel_offset,
            roi_rows=parallel_roi,
            roi_columns=serial_roi,
        )

    if serial_traps is not None:

        image = image.T.copy()

        image = _add_cti_to_image(
            image=image,
            clocker=serial_clocker,
            traps=serial_traps,
            ccd_volume=serial_ccd_volume,
            express=serial_express,
            offset=serial_offset,
            roi_rows=serial_roi,
            roi_columns=parallel_roi,
        )

        image = image.T

    return image


def remove_cti(
    image,
    iterations,
    parallel_express=0,
    parallel_clocker=None,
    parallel_ccd_volume=None,
    parallel_traps=None,
    parallel_offset=0,
    parallel_roi=None,
    serial_express=0,
    serial_clocker=None,
    serial_ccd_volume=None,
    serial_traps=None,
    serial_offset=0,
    serial_roi=None,
    prefill_traps=True,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns, for parallel and/or serial clocking.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values.
    iterations : int
        If CTI is being corrected, iterations determines the number of times 
        clocking is run to perform the correction via forward modeling. For 
        adding CTI only one run is required and iterations is ignored.
    parallel_express : int
        The factor by which pixel-to-pixel transfers are combined for 
        efficiency for parallel clocking.
    parallel_charge_injection_mode : bool
        If True, parallel clocking is performed in charge injection line 
        mode, where each pixel is clocked and therefore trailed by traps 
        over the entire CCD (as opposed to its distance from the CCD register).
    parallel_clocker : Clocker
        The object describing the clocking read-out electronics for parallel 
        clocking.
    parallel_ccd_volume : CCDVolume
        The object describing the CCD volume for parallel clocking. For 
        multi-phase clocking optionally use a list of different CCD volumes
        for each phase, in the same size list as parallel_clocker.sequence.
    parallel_traps : [Trap] or [[Trap]]
        A list of one or more trap objects for parallel clocking. To use 
        different types of traps that will require different watermark 
        levels, pass a 2D list of lists, i.e. a list containing lists of 
        one or more traps for each type.
    serial_* : *
        The same as the parallel_* objects described above but for serial 
        clocking instead.

    Returns
    -------
    image : np.ndarray
        The output array of pixel values with CTI removed.
    """

    # Initialise the iterative estimate of removed CTI
    image_remove_cti = deepcopy(image)

    # Estimate the image with removed CTI more precisely each iteration
    for iteration in range(iterations):

        image_add_cti = add_cti(
            image=image_remove_cti,
            parallel_express=parallel_express,
            parallel_clocker=parallel_clocker,
            parallel_ccd_volume=parallel_ccd_volume,
            parallel_traps=parallel_traps,
            parallel_offset=parallel_offset,
            parallel_roi=parallel_roi,
            serial_express=serial_express,
            serial_clocker=serial_clocker,
            serial_ccd_volume=serial_ccd_volume,
            serial_traps=serial_traps,
            serial_offset=serial_offset,
            serial_roi=serial_roi,
            prefill_traps=prefill_traps,
        )

        # Improved estimate of removed CTI
        image_remove_cti += image - image_add_cti

    return image_remove_cti
