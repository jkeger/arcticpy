""" AlgoRithm for Charge Transfer Inefficiency Correction

    Add or remove image trailing due to charge transfer inefficiency (CTI) in CCD detectors.
    
    https://github.com/jkeger/arcticpy

    Jacob Kegerreis (2020) jacob.kegerreis@durham.ac.uk
    James Nightingale
    Richard Massey

    WIP...
"""
import numpy as np
import math
import os
from copy import deepcopy

from arctic.roe import ROE
from arctic.ccd import ( CCD, CCDPhase )
from arctic.traps import (
    TrapLifetimeContinuum,
    TrapLogNormalLifetimeContinuum,
    TrapInstantCapture,
)
from arctic.trap_managers import (
    concatenate_trap_managers,
    TrapManager,
    TrapManagerTrackTime,
    TrapManagerInstantCapture,
)
from arctic import util


def express_matrix_from_rows_and_express(
    rows,
    express=0,
    dtype=float,
    offset=0,
    charge_injection=False,
    n_active_pixels=0,
    first_pixel_different=True,
):
    """ 
    To reduce runtime, instead of calculating the effects of every 
    pixel-to-pixel transfer, it is possible to approximate readout by 
    processing each transfer once (Anderson et al. 2010) or a few times 
    (Massey et al. 2014, section 2.1.5), then multiplying the effect of
    that transfer by the number of transfers it represents. This function
    computes the multiplicative factor, and returns it in a matrix that can 
    be easily looped over.

    Parameters
    ----------
    rows : int or list or range
        int:         The number of rows in an image.
        list/range:  A subset of rows in the image to be processed, for speed.
    express : int
        Parameter determining balance between accuracy (high values or zero)  
        and speed (low values).
            express = 0 (default, alias for express = n_pixels)
            express = 1 (fastest) --> compute the effect of each transfer once
            express = 2 --> compute n times the effect of those transfers that 
                      happen many times. After a few transfers (and e.g. eroded 
                      leading edges), the incremental effect of subsequent 
                      transfers can change.
            express = n_pixels (slowest) --> compute every pixel-to-pixel transfer
        Runtime scales as O(express^2).
    
    Optional parameters
    -------------------
    dtype : "int" or "float"
        Old versions of this algorithm assumed (unnecessarily) that all express 
        multipliers must be integers. It is slightly more efficient if this
        requirement is dropped, but the option to force it is included for
        backwards compatability.
    offset : int >=0
        Consider all pixels to be offset by this number of pixels from the 
        readout register. Useful if working out the matrix for a postage stamp 
        image, or to account for prescan pixels whose data is not stored.
    charge_injection : bool
        False: usual imaging operation of CCD, where (photo)electrons are 
               created in pixels then clocked through however many pixels lie 
               between it and the readout register.
        True:  electrons are electronically created by a charge injection 
               structure at the end of a CCD, then clocked through ALL of the 
               pixels to the readout register. By default, it is assumed that 
               this number is the number of rows in the image (but see n_rows).
    n_active_pixels : int
        Used only in charge injection mode, this specifies the number of pixels
        from the readout register to the farthest pixel. It is useful as a
        keyword if the image supplied is not the entire area, either because it
        is a postage stamp (whose final pixel is not the end of the image), or
        excludes pre-/over-scan pixels.
    first_pixel_different : bool
        Only used outside charge injection mode. Allows for the first
        pixel-to-pixel transfer differently to the rest. Physically, this may be
        because the first pixel that a charge cloud finds itself in is
        guaranteed to start with empty traps; whereas every other pixel's traps
        may  have been filled by other charge.
        In most situations, this makes it a factor ~(E+3)/(E+1) slower to run. 

    Returns
    -------
    express_matrix : [[float]]
        The express multiplier values for each pixel-to-pixel transfer.
    """

    # Parse inputs
    # Assumed number of rows in the (entire) image (even if given a postage stamp).
    #  with enough rows to contain the supposed image, including offset
    window = range(rows) if isinstance(rows, int) else rows
    n_rows = max(window) + 1 + offset
    # Default to very slow but accurate behaviour
    if express == 0:
        express = n_rows - offset
    if charge_injection:
        assert (
            first_pixel_different == True
        ), "Traps must be assumed initially empty in CIL mode (what else would they be filled with?)"
        n_rows = max(n_rows, n_active_pixels)
    elif first_pixel_different:
        n_rows -= 1

    # Initialise an array
    express_matrix = np.zeros((express, n_rows), dtype=dtype)

    # Compute the multiplier factors
    max_multiplier = n_rows / express
    # if it's going to be an integer, ceil() rather than int() is required in case the number of rows is not an integer multiple of express
    if dtype == int:
        max_multiplier = math.ceil(max_multiplier)
    if charge_injection:
        if dtype == int:
            for i in reversed(range(express)):
                express_matrix[i, :] = util.set_min_max(
                    max_multiplier, 0, n_rows - sum(express_matrix[:, 0])
                )
        else:
            express_matrix[:] = max_multiplier
        # if dtype == int:
        #    express_matrix[0,:] -= sum(express_matrix[:,0]) - n_rows
    else:
        # This populates every row in the matrix with a range from 1 to n_rows
        express_matrix[:] = np.arange(1, n_rows + 1)
        # Offset each row to account for the pixels that have already been read out
        for express_index in range(express):
            express_matrix[express_index] -= express_index * max_multiplier
        # Set all values to between 0 and max_multiplier
        express_matrix[express_matrix < 0] = 0
        express_matrix[express_matrix > max_multiplier] = max_multiplier

        # Separate out the first transfer of every pixel, so that it can be different from subsequent transfers (it sees only empty traps)
        if first_pixel_different:
            # What we calculated before contained (intentionally) one transfer too few. Store it...
            express_matrix_small = express_matrix
            # ..then create a new matrix with one extra transfer for each pixel
            n_rows += 1
            express_matrix = np.flipud(np.identity(n_rows))
            # Combine the two sets of transfers appropriately
            for i in range(express):
                n_nonzero = sum(express_matrix_small[i, :] > 0)
                express_matrix[n_nonzero, 1:] += express_matrix_small[i, :]

    # Extract the section of the array corresponding to the image (without the offset)
    n_rows -= offset
    express_matrix = express_matrix[:, offset:]  # remove the offset
    express_matrix = express_matrix[:, window]  # keep only the region of interest
    express_matrix = express_matrix[  # remove all rows containing only zeros, for speed later
        np.sum(express_matrix, axis=1) > 0, :
    ]

    # Adjust so that each computed transfer has equal effect (most useful if offset != 0). Could have got here directly.
    if dtype is not int:
        for i in range((express_matrix.shape)[1]):
            first_one = (np.where(express_matrix[:, i] == 1))[0]
            express_matrix[first_one, i] = 0
            nonzero = express_matrix[:, i] > 0
            if sum(nonzero) > 0:
                express_matrix[nonzero, i] = sum(express_matrix[nonzero, i]) / sum(
                    nonzero
                )
            express_matrix[first_one, i] = 1

    # Can't yet compute the when_to_store matrix, in case we are dealing with multi-phase readout
    return express_matrix


def _clock_charge_in_one_direction(
    image, roe, ccd, traps, express, offset, window_row, window_column,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values.
    roe : ROE
        The object describing the clocking of electrons with read-out 
        electronics.
    ccd : CCD
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

    # Parse inputs
    n_rows, n_columns = image.shape
    window_row = (
        range(n_rows) if window_row is None else window_row
    )  # list or range of which pixels to process in the redout direction
    window_column = (
        range(n_columns) if window_column is None else window_column
    )  # list or range of which pixels to process perpendicular to the readout direction
    if not isinstance(traps[0], list):
        traps = [
            traps
        ]  # If only a single trap species is used, still make sure it is an array
    phases_with_traps = [i for i,x in enumerate(ccd.fraction_of_traps) if x>0]
    steps_with_nonzero_dwell_time = [i for i,x in enumerate(roe.dwell_times) if x>0]

    # Calculate the number of times that the effect of each pixel-to-pixel transfer can be replicated
    express_matrix = express_matrix_from_rows_and_express(
        #dtype=int,
        rows=window_row,
        express=express,
        offset=offset,
        charge_injection=roe.charge_injection,
        first_pixel_different=roe.empty_traps_at_start,
    )
    (n_express, n_rows_to_clock) = express_matrix.shape
    assert n_rows_to_clock == n_rows, "n_rows_to_clock != n_rows"

    #
    # This is another way to achieve multi-step clocking, which omits a nested for loop and may be faster, but is less clear.
    #
    ## Prepare the image and express for multi-step clocking
    #n_steps_in_clock_cycle = len(roe.dwell_times)
    #assert n_steps_in_clock_cycle >= len(ccd.phase_fractional_widths)
    ##assert np.amax(ccd.phase_fractional_widths) <= 1
    #if n_steps_in_clock_cycle > 1:
    #    new_image = np.zeros((rows * n_steps_in_clock_cycle, columns))
    #    new_image[ccd.integration_phase :: n_steps_in_clock_cycle] = image
    #    image = new_image
    #    rows, columns = image.shape
    #    print(
    #        "Need to change window_row and window_column in case window is set; currently evaluating whole image"
    #    )
    #    window_row = range(n_rows_to_clock)
    #    window_column = range(n_columns)
    #    express_matrix = np.repeat(express_matrix, n_steps_in_clock_cycle, axis=1)


    # Set up an array of trap managers able to monitor the occupancy of (all types of) traps
    trap_managers = concatenate_trap_managers(traps, n_rows, ccd.fraction_of_traps)

    # Accounting for first transfer differently
    # Decide appropriate moments to store trap occupancy levels, so the next
    # EXPRESS iteration can continue from an (approximately) suitable configuration
    stored_trap_managers = trap_managers
    when_to_store_traps = np.zeros(express_matrix.shape, dtype=bool)
    if not roe.empty_traps_at_start and not roe.charge_injection:
        for express_index in range(n_express - 1):
            for row_index in range(n_rows - 1):
                if express_matrix[express_index + 1, row_index + 1] > 0:
                    break
            when_to_store_traps[express_index, row_index] = True

    
    # Expand image temporarily, if charge released from traps ever migrates to a different charge packet,
    # at any time during the clocking sequence
    n_rows_zero_padding = roe.max_referred_to_pixel - roe.min_referred_to_pixel
    if n_rows_zero_padding > 0:
        image = np.concatenate(
            (image, np.zeros((n_rows_zero_padding, n_columns), dtype=image.dtype)), axis=0
        ) 


    # Read out one column of pixels through one (column of) traps
    for column_index in range(len(window_column)):

        # Monitor the traps in every pixel, or just one (express=1) or a few
        # (express=a few) then replicate their effect
        for express_index in range(n_express):

            # Reset trap occupancy levels
            trap_managers = deepcopy(stored_trap_managers)

            # Each pixel
            for row_index in range(len(window_row)):
                
                express_multiplier = express_matrix[express_index, row_index]
                if express_multiplier == 0:
                    continue

                for clocking_step in steps_with_nonzero_dwell_time:
                    
                    for phase in phases_with_traps:

                        # Extract initial number of electrons from the relevant charge cloud
                        potential = roe.clock_sequence[clocking_step][phase]
                        row_read = window_row[row_index] + potential['capture_from_which_pixel']
                        n_free_electrons = image[row_read, window_column[column_index]] * potential['high']

                        # Allow electrons to be released from and captured by charge traps
                        n_electrons_released_and_captured = 0
                        n_electrons_trapped_before = 0
                        n_electrons_trapped_after = 0
                        for trap_manager in trap_managers[phase]:
                            n_electrons_trapped_before += trap_manager.n_trapped_electrons_from_watermarks(trap_manager.watermarks)
                            n_electrons_released_and_captured += trap_manager.n_electrons_released_and_captured(
                                n_free_electrons=n_free_electrons,
                                dwell_time=roe.dwell_times[clocking_step],
                                ccd=CCDPhase(ccd,phase),
                                express_multiplier=express_multiplier,
                                #ccd=ccd.extract_phase(phase),
                                #fractional_width=ccd.phase_fractional_widths[phase],
                            )
                            n_electrons_trapped_after += trap_manager.n_trapped_electrons_from_watermarks(trap_manager.watermarks)

                        # Return the released electrons back to the relevant charge cloud
                        
                        row_write = window_row[row_index] + potential['release_to_which_pixel']
                        image[row_write, window_column[column_index]] += (
                            n_electrons_released_and_captured * express_multiplier
                        )
                        print(row_index,'write row',row_write,'step',clocking_step,'phase',phase,
                              'n_ei',n_free_electrons,
                              'n_tb',n_electrons_trapped_before,
                              'n_ta',n_electrons_trapped_after,
                              'exp',express_multiplier,
                              'diff',(n_electrons_trapped_before-n_electrons_trapped_after),
                              'n_ef',image[row_write, window_column[column_index]])

                # Save trap occupancy at the end of one express
                if when_to_store_traps[express_index, row_index]:
                    stored_trap_managers = trap_managers
                    print('saving at', express_index, row_index)

        # Reset watermarks, effectively setting trap occupancy to zero
        if roe.empty_traps_between_columns: 
            [inner.empty_all_traps() for outer in stored_trap_managers for inner in outer]
        else:
            stored_trap_managers = deepcopy(trap_managers)

    # Recombine the image for multi-phase clocking
    #if n_simple_phases > 1:
    #    image = image.reshape((int(rows / n_simple_phases), n_simple_phases, columns)).sum(axis=1)
    # Unexpand image
    if n_rows_zero_padding > 0:
        image = image[ 0 : - n_rows_zero_padding, :]

    return image


def add_cti(
    image,
    parallel_express=0,
    parallel_roe=None,
    parallel_ccd=None,
    parallel_traps=None,
    parallel_offset=0,
    parallel_window=None,
    serial_express=0,
    serial_roe=None,
    serial_ccd=None,
    serial_traps=None,
    serial_offset=0,
    serial_window=None,
    prefill_traps=True,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns, for parallel and/or serial clocking.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values, assumed to be in units of electrons.
    parallel_express : int
        The factor by which pixel-to-pixel transfers are combined for 
        efficiency for parallel clocking.
    parallel_charge_injection_mode : bool
        If True, parallel clocking is performed in charge injection line 
        mode, where each pixel is clocked and therefore trailed by traps 
        over the entire CCD (as opposed to its distance from the CCD register).
    parallel_roe : ROE
        The object describing the clocking read-out electronics for parallel 
        clocking.
    parallel_ccd : CCD
        The object describing the CCD volume for parallel clocking. For 
        multi-phase clocking optionally use a list of different CCD volumes
        for each phase, in the same size list as parallel_roe.dwell_times.
    parallel_traps : [Trap] or [[Trap]]
        A list of one or more trap objects for parallel clocking. To use 
        different types of traps that will require different watermark 
        levels, pass a 2D list of lists, i.e. a list containing lists of 
        one or more traps for each type.
    parallel_offset : int
        The supplied image array is a postage stamp offset this number of 
        pixels from the readout register
    parallel_window : range() or list
        For speed, calculate only the effect on this subset of pixels. 
        Note that, because of edge effects, you should start the range several 
        pixels before the actual region of interest.
    serial_* : *
        The same as the parallel_* objects described above but for serial 
        clocking instead.

    Returns
    -------
    image : np.ndarray
        The output array of pixel values.
    """

    # If ROE not provided then assume simple, single-phase clocking
    if parallel_roe is None:
        parallel_roe = ROE()
    if serial_roe is None:
        serial_roe = ROE()

    # Don't modify the external array passed to this function
    image = deepcopy(image)

    if parallel_traps is not None:

        image = _clock_charge_in_one_direction(
            image=image,
            roe=parallel_roe,
            traps=parallel_traps,
            ccd=parallel_ccd,
            express=parallel_express,
            offset=parallel_offset,
            window_row=parallel_window,
            window_column=serial_window,
        )

    if serial_traps is not None:

        image = image.T.copy()

        image = _clock_charge_in_one_direction(
            image=image,
            roe=serial_roe,
            traps=serial_traps,
            ccd=serial_ccd,
            express=serial_express,
            offset=serial_offset,
            window_row=serial_window,
            window_column=parallel_window,
        )

        image = image.T

    return image


def remove_cti(
    image,
    iterations,
    parallel_express=0,
    parallel_roe=None,
    parallel_ccd=None,
    parallel_traps=None,
    parallel_offset=0,
    parallel_window=None,
    serial_express=0,
    serial_roe=None,
    serial_ccd=None,
    serial_traps=None,
    serial_offset=0,
    serial_window=None,
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
    parallel_roe : ROE
        The object describing the clocking read-out electronics for parallel 
        clocking.
    parallel_ccd : CCD
        The object describing the CCD volume for parallel clocking. For 
        multi-phase clocking optionally use a list of different CCD volumes
        for each phase, in the same size list as parallel_roe.dwell_times.
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
            parallel_roe=parallel_roe,
            parallel_ccd=parallel_ccd,
            parallel_traps=parallel_traps,
            parallel_offset=parallel_offset,
            parallel_window=parallel_window,
            serial_express=serial_express,
            serial_roe=serial_roe,
            serial_ccd=serial_ccd,
            serial_traps=serial_traps,
            serial_offset=serial_offset,
            serial_window=serial_window,
            prefill_traps=prefill_traps,
        )

        # Improved estimate of removed CTI
        image_remove_cti += image - image_add_cti

    return image_remove_cti
