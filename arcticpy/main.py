import numpy as np
from copy import deepcopy

from arcticpy.roe import ROE
from arcticpy.ccd import CCD, CCDPhase
from arcticpy.trap_managers import AllTrapManager
from arcticpy import util


def _clock_charge_in_one_direction(
    image,
    roe,
    ccd,
    traps,
    express,
    offset,
    window_row_range,
    window_column_range,
    time_window_express_range,
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons 
    along their independent columns.

    Parameters
    ----------
    image : [[float]]
        The input array of pixel values, assumed to be in units of electrons.
        
        The first dimension is the "row" index, the second is the "column" 
        index. By default (for parallel clocking), charge is transfered "up" 
        from row n to row 0 along each independent column. i.e. the readout 
        register is above row 0. (For serial clocking, the image is rotated 
        beforehand, outside of this function, see add_cti().)
        
        e.g. (with arbitrary trap parameters)
        Initial image with one bright pixel in the first three columns:
            [[0.0,     0.0,     0.0,     0.0  ], 
             [200.0,   0.0,     0.0,     0.0  ], 
             [0.0,     200.0,   0.0,     0.0  ], 
             [0.0,     0.0,     200.0,   0.0  ], 
             [0.0,     0.0,     0.0,     0.0  ], 
             [0.0,     0.0,     0.0,     0.0  ]]
        Final image with CTI trails behind each bright pixel:
            [[0.0,     0.0,     0.0,     0.0  ], 
             [196.0,   0.0,     0.0,     0.0  ], 
             [3.0,     194.1,   0.0,     0.0  ], 
             [2.0,     3.9,     192.1,   0.0  ], 
             [1.3,     2.5,     4.8,     0.0  ], 
             [0.8,     1.5,     2.9,     0.0  ]]
        
    roe : ROE
        An object describing the timing and direction(s) in which electrons are 
        moved during readout.
        
    ccd : CCD
        An object describing the way in which a cloud of electrons fills the CCD 
        volume. 
        
    traps : [Trap] or [[Trap]]
        A list of one or more trap objects. To use different types of traps that
        will require different watermark levels, pass a 2D list of lists, i.e. a 
        list containing lists of one or more traps for each type. 
        
    express : int
        The number of times the pixel-to-pixel transfers are computed, 
        determining the balance between accuracy (high values) and speed 
        (low values) (Massey et al. 2014, section 2.1.5).
            n_pix   (slower, accurate) Compute every pixel-to-pixel 
                    transfer. The default 0 = alias for n_pix.
            k       Recompute on k occasions the effect of each transfer.  
                    After a few transfers (and e.g. eroded leading edges),  
                    the incremental effect of subsequent transfers can change.
            1       (faster, approximate) Compute the effect of each 
                    transfer only once.
        Runtime scales approximately as O(express^0.5). ###WIP
        
    offset : int (>= 0)
        The number of (e.g. prescan) pixels separating the supplied image from 
        the readout register.
        
    window_row_range : range
        The subset of row pixels to model, to save time when only a specific 
        region of the image is of interest. Defaults to range(0, n_pixels) for 
        the full image.
    
    window_column_range : range
        The subset of column pixels to model, to save time when only a specific 
        region of the image is of interest. Defaults to range(0, n_columns) for 
        the full image.
        
    time_window_express_range : range
        The subset of transfers to implement. Defaults to range(0, n_pixels) for 
        the full image.
        
        The entire readout is still modelled, but only the results from this 
        subset of transfers are implemented in the final image.

    Returns
    -------
    image : [[float]]
        The output array of pixel values.
    """

    # Generate the arrays over each step for: the number of of times that the
    # effect of each pixel-to-pixel transfer can be multiplied for the express
    # algorithm; and whether the traps must be monitored (usually whenever
    # express matrix > 0)
    (
        express_matrix,
        monitor_traps_matrix,
    ) = roe.express_matrix_and_monitor_traps_matrix_from_pixels_and_express(
        pixels=window_row_range,
        express=express,
        offset=offset,
        time_window_express_range=time_window_express_range,
    )
    # ; and whether the trap occupancy states must be saved for the next express
    # pass rather than being reset (usually at the end of each express pass)
    save_trap_states_matrix = roe.save_trap_states_matrix_from_express_matrix(
        express_matrix=express_matrix
    )

    n_express, n_rows_to_process = express_matrix.shape

    # Decide in advance which steps need to be evaluated and which can be skipped
    phases_with_traps = [
        i for i, frac in enumerate(ccd.fraction_of_traps_per_phase) if frac > 0
    ]
    steps_with_nonzero_dwell_time = [
        i for i, time in enumerate(roe.dwell_times) if time > 0
    ]

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

    # Read out one column of pixels through the (column of) traps
    for column_index in range(len(window_column_range)):

        # Monitor the traps in every pixel, or just one (express=1) or a few
        # (express=a few) then replicate their effect
        for express_index in range(n_express):
            # Restore the trap occupancy levels (to empty, or to a saved state
            # from a previous express pass)
            trap_managers.restore()

            # Each pixel
            for row_index in range(len(window_row_range)):
                express_multiplier = express_matrix[express_index, row_index]
                # Skip this step if not needed to be evaluated
                if not monitor_traps_matrix[express_index, row_index]:
                    continue

                for clocking_step in steps_with_nonzero_dwell_time:
                    n_electrons_trapped = 0

                    for phase in phases_with_traps:
                        # Information about the potentials in this phase
                        roe_phase = roe.clock_sequence[clocking_step][phase]

                        # Select the relevant pixel (and phase) for the initial charge
                        row_index_read = (
                            window_row_range[row_index]
                            + roe_phase.capture_from_which_pixels
                        )

                        # Initial charge (0 if this phase's potential is not high)
                        n_free_electrons = (
                            image[row_index_read, window_column_range[column_index]]
                            * roe_phase.is_high
                        )

                        # Allow electrons to be released from and captured by traps
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

                        # Select the relevant pixel (and phase) for the returned charge
                        row_index_write = (
                            window_row_range[row_index]
                            + roe_phase.release_to_which_pixels
                        )

                        # Return the electrons back to the relevant charge
                        # cloud, or a fraction if they are being returned to
                        # multiple phases
                        image[row_index_write, window_column_range[column_index]] += (
                            n_electrons_released_and_captured
                            * roe_phase.release_fraction_to_pixel
                            * express_multiplier
                        )

                # Save the trap occupancy states if required for the next express pass
                if save_trap_states_matrix[express_index, row_index]:
                    trap_managers.save()

        # Reset the watermarks, effectively setting the trap occupancies to zero
        if roe.empty_traps_between_columns:
            trap_managers.empty_all_traps()
        trap_managers.save()

    # Unexpand the image to its original dimensions
    if n_rows_zero_padding > 0:
        image = image[0:-n_rows_zero_padding, :]

    return image


def add_cti(
    image,
    parallel_ccd=None,
    parallel_roe=None,
    parallel_traps=None,
    parallel_express=0,
    parallel_offset=0,
    parallel_window_range=None,
    serial_ccd=None,
    serial_roe=None,
    serial_traps=None,
    serial_express=0,
    serial_offset=0,
    serial_window_range=None,
    time_window=[0, 1],
):
    """
    Add CTI trails to an image by trapping, releasing, and moving electrons
    along their independent columns, for parallel and/or serial clocking.

    Parameters
    ----------
    image : [[float]]
        The input array of pixel values, assumed to be in units of electrons.
        
        The first dimension is the "row" index, the second is the "column" 
        index. By default (for parallel clocking), charge is transfered "up" 
        from row n to row 0 along each independent column. i.e. the readout 
        register is above row 0. (For serial clocking, the image is rotated 
        before modelling, such that charge moves from column n to column 0.)
        
        e.g. (with arbitrary trap parameters)
        Initial image with one bright pixel in the first three columns:
            [[0.0,     0.0,     0.0,     0.0  ], 
             [200.0,   0.0,     0.0,     0.0  ], 
             [0.0,     200.0,   0.0,     0.0  ], 
             [0.0,     0.0,     200.0,   0.0  ], 
             [0.0,     0.0,     0.0,     0.0  ], 
             [0.0,     0.0,     0.0,     0.0  ]]
        Image with parallel CTI trails:
            [[0.0,     0.0,     0.0,     0.0  ], 
             [196.0,   0.0,     0.0,     0.0  ], 
             [3.0,     194.1,   0.0,     0.0  ], 
             [2.0,     3.9,     192.1,   0.0  ], 
             [1.3,     2.5,     4.8,     0.0  ], 
             [0.8,     1.5,     2.9,     0.0  ]]
        Final image with parallel and serial CTI trails:
            [[0.0,     0.0,     0.0,     0.0  ], 
             [194.1,   1.9,     1.5,     0.9  ], 
             [2.9,     190.3,   2.9,     1.9  ], 
             [1.9,     3.8,     186.5,   3.7  ], 
             [1.2,     2.4,     4.7,     0.9  ], 
             [0.7,     1.4,     2.8,     0.6  ]]
        
    parallel_express : int
        The number of times the transfers are computed, determining the 
        balance between accuracy (high values) and speed (low values), for 
        parallel clocking (Massey et al. 2014, section 2.1.5).
        
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
        
    parallel_offset : int (>= 0)
        The number of (e.g. prescan) pixels separating the supplied image from 
        the readout register. i.e. Treat the input image as a sub-image that is 
        offset this number of pixels from readout, increasing the number of
        pixel-to-pixel transfers.
        
    parallel_window_range : range
        For speed, calculate only the effect on this subset of pixels. Defaults
        to range(0, n_pixels) for the full image.
        
        For a single pixel (e.g. for trap pumping), can enter just the single 
        integer index, which will be converted to range(index, index + 1).
        
        Note that, because of edge effects, you should start the range several
        pixels before the actual region of interest.
        
    serial_* : *
        The same as the parallel_* objects described above but for serial
        clocking instead.
        
    time_window : [float, float]
        The beginning and end of the time during readout to be implemented, as a 
        fraction of the total readout from 0 to 1. 
        
        ### Would a transfer/pixel range be better than fractions? Maybe made 
        too complicated by offsets etc?
        
        The entire readout is still modelled, but only the results from this 
        subset of transfers are implemented in the final image.
        
        This could be used to e.g. add cosmic rays during readout of simulated
        images. Successive calls to complete the readout should start at
        the same value that the previous one ended, e.g. [0, 0.5] then
        [0.5, 1]. Be careful not to divide the readout too finely, as there is
        only as much temporal resolution as there are rows (not rows * phases)
        in the image. Also, for each time that readout is split between
        successive calls to this function, the output in one row of pixels
        will change slightly (unless express=0) because trap occupancy is
        not stored between calls.

    Returns
    -------
    image : [[float]]
        The output array of pixel values.
    """
    n_rows_in_image, n_columns_in_image = image.shape

    # Default windows to the full image; convert single-pixel windows to ranges
    if parallel_window_range is None:
        parallel_window_range = range(n_rows_in_image)
    elif isinstance(parallel_window_range, int):
        parallel_window_range = range(parallel_window_range, parallel_window_range + 1)
    if serial_window_range is None:
        serial_window_range = range(n_columns_in_image)
    elif isinstance(serial_window_range, int):
        serial_window_range = range(serial_window_range, serial_window_range + 1)

    # Set the express range for the time window; default to the full image
    if time_window == [0, 1]:
        time_window_express_range = None
        # ###
        serial_window_column_range = parallel_window_range
    else:
        time_window_express_range = range(
            int(time_window[0] * (n_rows_in_image + parallel_offset)),
            int(time_window[1] * (n_rows_in_image + parallel_offset)),
        )
        if len(time_window_express_range) == 0:
            serial_window_column_range = range(0)
        else:
            # Intersection of spatial and temporal windows of interest
            serial_window_column_range = range(
                max(
                    parallel_window_range[0],
                    time_window_express_range[0] - parallel_offset,
                ),
                min(
                    parallel_window_range[-1],
                    time_window_express_range[-1] - parallel_offset,
                )
                + 1,
            )

    # Default ROE: simple, single-phase clocking in imaging mode
    if parallel_roe is None:
        parallel_roe = ROE()
    if serial_roe is None:
        serial_roe = ROE()

    # Don't modify the external array passed to this function
    image = deepcopy(image)

    # Parallel clocking
    if parallel_traps is not None:

        # Transfer charge in parallel direction
        image = _clock_charge_in_one_direction(
            image=image,
            ccd=parallel_ccd,
            roe=parallel_roe,
            traps=parallel_traps,
            express=parallel_express,
            offset=parallel_offset,
            window_row_range=parallel_window_range,
            window_column_range=serial_window_range,
            time_window_express_range=time_window_express_range,
        )

    # Serial clocking
    if serial_traps is not None:

        # Switch axes, so clocking happens in other direction
        image = image.T.copy()

        # Transfer charge in serial direction
        image = _clock_charge_in_one_direction(
            image=image,
            ccd=serial_ccd,
            roe=serial_roe,
            traps=serial_traps,
            express=serial_express,
            offset=serial_offset,
            window_row_range=serial_window_range,
            window_column_range=serial_window_column_range,
            time_window_express_range=None,
        )

        # Switch axes back
        image = image.T

    return image


def remove_cti(
    image,
    iterations,
    parallel_ccd=None,
    parallel_roe=None,
    parallel_traps=None,
    parallel_express=0,
    parallel_offset=0,
    parallel_window_range=None,
    serial_ccd=None,
    serial_roe=None,
    serial_traps=None,
    serial_express=0,
    serial_offset=0,
    serial_window_range=None,
    time_window=[0, 1],
):
    """
    Remove CTI trails from an image by first modelling the addition of CTI.
    
    See add_cti()'s documentation for the forward modelling. This function 
    iteratively models the addition of more CTI trails to the input image to 
    then extract the corrected image without the original trails.

    Parameters
    ----------
    All parameters are identical to those of add_cti() as described in its 
    documentation, with the exception of:
    
    iterations : int
        The number of times CTI-adding clocking is run to perform the correction 
        via forward modelling. 

    Returns
    -------
    image : [[float]]
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
            parallel_window_range=parallel_window_range,
            serial_ccd=serial_ccd,
            serial_roe=serial_roe,
            serial_traps=serial_traps,
            serial_express=serial_express,
            serial_offset=serial_offset,
            serial_window_range=serial_window_range,
            time_window=time_window,
        )

        # Improved estimate of image with CTI trails removed
        image_remove_cti += image - image_add_cti

    return image_remove_cti
