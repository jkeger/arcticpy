"""
arCTIc: algoRithm for Charge Transfer Inefficiency correction

Add or remove image trailing due to charge transfer inefficiency (CTI) in CCD detectors.
    
https://github.com/jkeger/arcticpy

Jacob Kegerreis (2020) jacob.kegerreis@durham.ac.uk
Richard Massey r.j.massey@durham.ac.uk
James Nightingale
"""
import numpy as np
from copy import deepcopy

from arcticpy.roe import ROE
from arcticpy.ccd import CCD, CCDPhase
from arcticpy.traps import (
    Trap,
    TrapLifetimeContinuum,
    TrapLogNormalLifetimeContinuum,
    TrapInstantCapture,
)
from arcticpy.trap_managers import AllTrapManager
from arcticpy import util


def _clock_charge_in_one_direction(
    image, roe, ccd, traps, express, offset, window_row, window_column, window_express
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns.

    Parameters
    ----------
    image : np.ndarray
        The input array of pixel values.
    roe : ROE
        An object describing the timing and direction(s) in which electrons are moved
        during readout
    ccd : CCD
        An object describing the way in which a cloud of electrons fills the CCD volume. 
    traps : [Trap] or [[Trap]]
        A list of one or more trap objects. To use different types of traps 
        that will require different watermark levels, pass a 2D list of 
        lists, i.e. a list containing lists of one or more traps for each 
        type. 
    express : int
        The factor by which pixel-to-pixel transfers are combined for efficiency.
    offset : int
        The number of (e.g. prescan) pixels separating the supplied image from the readout 
        node.

    Returns
    -------
    image : np.ndarray
        The output array of pixel values.
    """

    # Calculate the number of times that the effect of each pixel-to-pixel transfer can be replicated
    express_matrix, when_to_monitor_traps, when_to_store_traps = roe.express_matrix_from_pixels_and_express(
        window_row, express=express, offset=offset, window_express=window_express
    )
    (n_express, n_rows_to_process) = express_matrix.shape
    
    # Decide in advance which steps need to be evaluated, and which can be skipped
    phases_with_traps = [i for i, x in enumerate(ccd.fraction_of_traps) if x > 0]
    steps_with_nonzero_dwell_time = [i for i, x in enumerate(roe.dwell_times) if x > 0]
    express_with_nonzero_effect = (np.sum(express_matrix, axis=1) > 0).nonzero()

    # Set up an array of trap managers able to monitor the occupancy of (all types of) traps
    max_n_transfers = n_rows_to_process * len(steps_with_nonzero_dwell_time)
    trap_managers = AllTrapManager(traps, max_n_transfers, ccd)

    # Temporarily expand image, if charge released from traps ever migrates to
    # a different charge packet, at any time during the clocking sequence
    n_rows_zero_padding = max(roe.pixels_accessed_during_clocking) - min(
        roe.pixels_accessed_during_clocking
    )
    zero_padding = np.zeros((n_rows_zero_padding, image.shape[1]), dtype=image.dtype)
    image = np.concatenate((image, zero_padding), axis=0)

    # Read out one column of pixels through one (column of) traps
    for column_index in range(len(window_column)):

        # Monitor the traps in every pixel, or just one (express=1) or a few
        # (express=a few) then replicate their effect
        for express_index in range(n_express):

            # Reset trap occupancy levels
            trap_managers.restore()
            checksum = np.sum(image[:, window_column[column_index]])

            # Each pixel
            for row_index in range(len(window_row)):

                express_multiplier = express_matrix[express_index, row_index]
                #if express_multiplier == 0:
                if not when_to_monitor_traps[express_index, row_index]:
                    continue

                for clocking_step in steps_with_nonzero_dwell_time:
                    n_electrons_trapped = 0

                    for phase in phases_with_traps:

                        # Extract initial number of electrons from the relevant charge cloud
                        potential = roe.clock_sequence[clocking_step][phase]
                        row_read = (
                            window_row[row_index]
                            + potential["capture_from_which_pixel"]
                        )
                        n_free_electrons = (
                            image[row_read, window_column[column_index]]
                            * potential["high"]
                        )

                        # Allow electrons to be released from and captured by charge traps
                        n_electrons_released_and_captured = 0
                        for trap_manager in trap_managers[phase]:
                            n_electrons_released_and_captured += trap_manager.n_electrons_released_and_captured(
                                n_free_electrons=n_free_electrons,
                                dwell_time=roe.dwell_times[clocking_step],
                                ccd_filling_function=ccd.cloud_fractional_volume_from_n_electrons_in_phase(
                                    phase
                                ),
                                express_multiplier=express_multiplier,
                            )

                        # Return the released electrons back to the relevant charge cloud
                        row_write = (
                            window_row[row_index] + potential["release_to_which_pixel"]
                        )
                        image[row_write, window_column[column_index]] += (
                            n_electrons_released_and_captured
                            * potential["release_fraction_to_pixel"]
                            * express_multiplier
                        )

                # At end of (each express pass on) each row, check whether trap occupancy will be required for next express pass
                if when_to_store_traps[express_index, row_index]:
                    trap_managers.save()

        # At end of each column, reset watermarks, effectively setting trap occupancy to zero
        if roe.empty_traps_between_columns:
            trap_managers.empty_all_traps()
        trap_managers.save()

    # Unexpand image
    if n_rows_zero_padding > 0:
        image = image[0:-n_rows_zero_padding, :]

    return image


def add_cti(
    image,
    parallel_roe=None,
    parallel_ccd=None,
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
    time_window=[0,1],
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
        pixels from the readout register. This increases the number of
        pixel-to-pixel transfers assumed if readout is normal (and has no
        effect for other types of clocking).
    parallel_window : range()
        For speed, calculate only the effect on this subset of pixels.
        Note that, because of edge effects, you should start the range several
        pixels before the actual region of interest.
    serial_* : *
        The same as the parallel_* objects described above but for serial
        clocking instead.
    time_window : [float, float]
        The beginning and end of the time during readout to be calculated.
        This could be used to add cosmic rays during readout of simulated
        images. Successive calls to complete the readout should start at
        the same value that the previous one ended, e.g. [0,0.5] then
        [0.5,1]. Be careful not to divide readout too finely, as there is
        only as much temporal resolution as there are rows (not rows * phases)
        in the image. Also, for each time that readout is split between
        successive calls to this function, the output in one row of pixels
        will change slightly (unless express=0) because trap occupancy is
        not stored between calls.

    Returns
    -------
    image : np.ndarray
        The output array of pixel values.
    """

    # Parse inputs
    n_rows_in_image, n_columns_in_image = image.shape
    if parallel_window is None:
        parallel_window = range(n_rows_in_image)
    elif isinstance(parallel_window, int):
        parallel_window = range(parallel_window, parallel_window + 1)
    if serial_window is None:
        serial_window = range(n_columns_in_image)
    elif isinstance(serial_window, int):
        serial_window = range(serial_window, serial_window + 1)
    if time_window == [0,1]:
        express_window = None
        window_column_serial = parallel_window
    else:
        express_window = range(
            int(time_window[0] * (n_rows_in_image + parallel_offset)),
            int(time_window[1] * (n_rows_in_image + parallel_offset))
        )
        if len(express_window) == 0:
            window_column_serial = range(0)
        else:
            window_column_serial = range(  # intersection of spatial and temporal windows of interest
                max(parallel_window[0], express_window[0] - parallel_offset),
                min(parallel_window[-1], express_window[-1] - parallel_offset) + 1
            )

    # If ROE not provided then assume simple, single-phase clocking in imaging mode
    if parallel_roe is None:
        parallel_roe = ROE()
    if serial_roe is None:
        serial_roe = ROE()

    # Don't modify the external array passed to this function
    image = deepcopy(image)

    # Parallel clocking
    if parallel_traps is not None:

        image = _clock_charge_in_one_direction(
            image=image,
            ccd=parallel_ccd,
            roe=parallel_roe,
            traps=parallel_traps,
            express=parallel_express,
            offset=parallel_offset,
            window_row=parallel_window,
            window_column=serial_window,
            window_express=express_window,
        )

    # Serial clocking
    if serial_traps is not None:

        # Switch axes, so clocking happens in other direction
        image = image.T.copy()

        # Read out charge in serial direction
        image = _clock_charge_in_one_direction(
            image=image,
            ccd=serial_ccd,
            roe=serial_roe,
            traps=serial_traps,
            express=serial_express,
            offset=serial_offset,
            window_row=serial_window,
            window_column=window_column_serial,
            window_express=None,
        )

        # Switch axes back
        image = image.T

    # Finished
    return image


def remove_cti(
    image,
    iterations,
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
    time_window=[0,1]
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
        pixels from the readout register. This increases the number of
        pixel-to-pixel transfers assumed if readout is normal (and has no
        effect for other types of clocking).
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
        The output array of pixel values with CTI removed.
    """

    # Initialise the iterative estimate of removed CTI; don't modify the external array
    image_remove_cti = deepcopy(image)

    # Estimate the image with removed CTI more precisely each iteration
    for iteration in range(iterations):

        image_add_cti = add_cti(
            image=image_remove_cti,
            parallel_ccd=parallel_ccd,
            parallel_roe=parallel_roe,
            parallel_traps=parallel_traps,
            parallel_express=parallel_express,
            parallel_offset=parallel_offset,
            parallel_window=parallel_window,
            serial_ccd=serial_ccd,
            serial_roe=serial_roe,
            serial_traps=serial_traps,
            serial_express=serial_express,
            serial_offset=serial_offset,
            serial_window=serial_window,
            time_window=time_window,
        )

        # Improved estimate of image with CTI trails removed
        image_remove_cti += image - image_add_cti

    # Finished
    return image_remove_cti
