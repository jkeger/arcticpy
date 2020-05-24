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
    (Massey et al. 2014, section 2.1.5), then multiplying their effects.
    This function computes the multiplicative factor, in a matrix that can
    be easily looped over.

    Parameters
    ----------
    rows : int or list or range
        int:         The number of rows in an image.
        list/range:  A subset of rows in the image to be processed, for speed.
    express : int
        Parameter determining balance between accuracy (high values or zero)  
        and speed (low values).
            express = 0 (default, slow) --> compute every pixel-to-pixel transfer
            express = 1 (fastest) --> compute the effect of each transfer once
            express = 2 --> compute n times the effect of those transfers that 
                      happen many times. After a few transfers (and e.g. eroded 
                      leading edges), the incremental effect of subsequent 
                      transfers can change.
    
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

If set to  In most situations, this is a factor ~(E+3)/(E+1) slower to run. 

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

    # Adjust so that each computed transfer has equal effect (most useful if offset <> 0). Could have got here directly.
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


def _clock_charge_in_one_dimension(
    image, roe, ccd, traps, express, offset, window_readout, window_across,
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
    window_readout = (
        range(n_rows) if window_readout is None else window_readout
    )  # list or range of which pixels to process in the redout direction
    window_across = (
        range(n_columns) if window_across is None else window_across
    )  # list or range of which pixels to process perpendicular to the readout direction
    if not isinstance(traps[0], list):
        traps = [
            traps
        ]  # If only a single trap species is used, still make sure it is an array

    # Calculate the number of times that the effect of each pixel-to-pixel transfer can be replicated
    express_matrix = express_matrix_from_rows_and_express(
        # dtype=int,
        rows=window_readout,
        express=express,
        offset=offset,
        charge_injection=roe.charge_injection,
        first_pixel_different=roe.empty_traps_at_start,
    )
    (n_express, n_rows) = express_matrix.shape

    #
    # This is another way to achieve multi-step clocking, which omits a nested for loop and may be faster, but is less clear.
    #
    ## Prepare the image and express for multi-step clocking
    n_steps_in_clock_cycle = len(roe.dwell_times)
    #assert n_steps_in_clock_cycle >= len(ccd.phase_fractional_widths)
    ##assert np.amax(ccd.phase_fractional_widths) <= 1
    #if n_steps_in_clock_cycle > 1:
    #    new_image = np.zeros((rows * n_steps_in_clock_cycle, columns))
    #    new_image[ccd.integration_phase :: n_steps_in_clock_cycle] = image
    #    image = new_image
    #    rows, columns = image.shape
    #    print(
    #        "Need to change window_readout and window_across in case window is set; currently evaluating whole image"
    #    )
    #    window_readout = range(n_rows)
    #    window_across = range(n_columns)
    #    express_matrix = np.repeat(express_matrix, n_steps_in_clock_cycle, axis=1)

    # Set up an array of trap managers able to monitor the occupancy of all
    # (types of) traps in (one phase of) a pixel
    # NB: these are automatically created with all traps empty
    trap_managers_one_phase = []
    for trap_group in traps:
        if trap_group[0].discrete:
            for trap in trap_group:
                trap_managers_one_phase.append(TrapManagerDiscrete(traps=trap, rows=rows))               
        # Use a non-default trap manager if required for the input trap species
        elif isinstance(
            trap_group[0], (TrapLifetimeContinuum, TrapLogNormalLifetimeContinuum),
        ):
            trap_managers_one_phase.append(TrapManagerTrackTime(traps=trap_group, rows=n_rows))
        elif isinstance(trap_group[0], TrapInstantCapture):
            trap_managers_one_phase.append(TrapManagerInstantCapture(traps=trap_group, rows=n_rows))
        else:
            trap_managers_one_phase.append(TrapManager(traps=trap_group, rows=n_rows))

   
    # Replicate trap managers to keep track of traps in different phases separately
    trap_managers = []
    for _ in range(ccd.n_phases): trap_managers.append(trap_managers_one_phase)
    phases_with_traps = [i for i,x in enumerate(ccd.fraction_of_traps_per_phase) if x>0]
 
    
#    # Replicate trap managers to keep track of traps in different phases separately
#    trap_managers = []
#    for i in range(ccd.n_phases): 
#        trap_managers_this_phase = deepcopy(trap_managers_one_phase)
#        #print(trap_managers_this_phase.densities)
#        #
#        # Caution; next line makes densities differe from those in manager.traps.density - should add setter and getter to trap_manager
#        #
#        #trap_managers_this_phase.densities *= ccd.fraction_of_traps_per_phase[i]
#        trap_managers.append(trap_managers_this_phase)
#    phases_with_traps = [i for i,x in enumerate(ccd.fraction_of_traps_per_phase) if x>0]
    
    # Store trap managers with empty traps, for future recall
    rowwise_stored_trap_managers = trap_managers
    columnwise_stored_trap_managers = trap_managers

    # Accounting for first transfer differently
    # Decide appropriate moments to store trap occupancy levels, so the next
    # EXPRESS iteration can continue from an (approximately) suitable configuration
    when_to_store_traps = np.zeros(express_matrix.shape, dtype=bool)
    if not roe.empty_traps_at_start and not roe.charge_injection:
        for express_index in range(n_express - 1):
            for row_index in range(n_rows - 1):
                if express_matrix[express_index + 1, row_index] > 0:
                    break
            when_to_store_traps[express_index, row_index] = True
    

    # Read out one column of pixels through one (column of) traps
    for column_index in range(len(window_across)):
    
        # Reset watermarks, effectively setting trap occupancy to zero
        trap_managers = deepcopy(columnwise_stored_trap_managers)
        #for trap_manager in trap_managers:
        #    trap_manager.empty_all_traps()

        # Monitor the traps in every pixel, or just one (express=1) or a few
        # (express=a few) then replicate their effect
        for express_index in range(n_express):

            # Reset trap occupancy levels for next express loop
            trap_managers = deepcopy(rowwise_stored_trap_managers)

            # Each pixel
            for row_index in range(len(window_readout)):
                express_multiplier = express_matrix[express_index, row_index]
                if express_multiplier == 0:
                    continue
                step_in_clock_cycle = row_index % n_steps_in_clock_cycle
                step_in_clock_cycle = 0
                dwell_time = roe.dwell_times[step_in_clock_cycle]

                # Save trap occupancy
                if when_to_store_traps[express_index, row_index]:
                    rowwise_stored_trap_managers = trap_managers

                #for clocking_step in clocking_sequence:
                for phase in phases_with_traps:
                
                    # Initial number of electrons available for trapping
                    n_free_electrons = image[
                        window_readout[row_index], window_across[column_index]
                    ]

                    # Release and capture
                    n_electrons_released_and_captured = 0
                    
                    #trap_managers_one_phase=trap_managers[phase]
                    
                    for trap_manager in trap_managers[phase]:
                        n_electrons_released_and_captured += trap_manager.n_electrons_released_and_captured(
                            n_free_electrons=n_free_electrons,
                            dwell_time=dwell_time,
                            ccd=CCDPhase(ccd,phase),
                            #ccd=ccd.extract_phase(phase),
                            fractional_width=ccd.phase_fractional_widths[phase],
                            express_multiplier=express_multiplier,
                        )

                    # Total change to electrons in pixel
                    n_free_electrons += (
                        n_electrons_released_and_captured * express_multiplier
                    )

                    image[
                        window_readout[row_index], window_across[column_index]
                    ] = n_free_electrons

    # Recombine the image for multi-phase clocking
    #if n_simple_phases > 1:
    #    image = image.reshape((int(rows / n_simple_phases), n_simple_phases, columns)).sum(axis=1)

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

        image = _clock_charge_in_one_dimension(
            image=image,
            roe=parallel_roe,
            traps=parallel_traps,
            ccd=parallel_ccd,
            express=parallel_express,
            offset=parallel_offset,
            window_readout=parallel_window,
            window_across=serial_window,
        )

    if serial_traps is not None:

        image = image.T.copy()

        image = _clock_charge_in_one_dimension(
            image=image,
            roe=serial_roe,
            traps=serial_traps,
            ccd=serial_ccd,
            express=serial_express,
            offset=serial_offset,
            window_readout=serial_window,
            window_across=parallel_window,
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
