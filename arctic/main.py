""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy

from arctic.clock import Clocker
from arctic.traps import (
    TrapNonUniformHeightDistribution,
    TrapManager,
    TrapManagerNonUniformHeightDistribution,
)


class ArcticMain(object):
    def __init__(
        self,
        iterations=1,
        parallel_express=0,
        parallel_charge_injection_mode=False,
        serial_express=0,
    ):
        """
        The primary class for the adding or removal of CTI, using the parameters 
        provided by the other input class objects.
        
        Parameters
        ----------
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
        serial_express : int
            The factor by which pixel-to-pixel transfers are combined for 
            efficiency for serial clocking.
        """

        self.iterations = iterations

        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.serial_express = serial_express

    @staticmethod
    def express_matrix_from_rows_and_express(rows, express):
        """ 
        Calculate the values by which the effects of clocking each pixel must be multiplied when using the express 
        algorithm. See Massey et al. (2014), section 2.1.5.

        Parameters
        --------
        express : int
            The express parameter = (maximum number of transfers) / (number of computed tranfers), i.e. the number of 
            computed transfers per pixel. So express = 1 --> maxmimum speed; express = (maximum number of transfers) 
            --> maximum accuracy (Must be a factor of rows).
        rows : int
            The number of rows in the CCD and hence the maximum number of transfers.

        Returns
        --------
        express_multiplier : [[float]]
            The express multiplier values for each pixel.
        """

        express = rows if express == 0 else express

        # Initialise the array
        express_multiplier = np.empty((express, rows), dtype=int)
        express_multiplier[:] = np.arange(1, rows + 1)

        express_max = int(rows / express)

        # Offset each row to account for the pixels that have already been read out
        for express_index in range(express):
            express_multiplier[express_index] -= express_index * express_max

        # Set all values to between 0 and express_max
        express_multiplier[express_multiplier < 0] = 0
        express_multiplier[express_max < express_multiplier] = express_max

        return express_multiplier

    def _add_cti_to_image(self, image, clocker, traps, ccd_volume, express):
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
        traps : [Trap] or [[Trap]]
            A list of one or more trap objects. To use different types of traps 
            that will require different watermark levels, pass a 2D list of 
            lists, i.e. a list containing lists of one or more traps for each 
            type. 
        ccd_volume : CCDVolume
            The object describing the CCD volume. 
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
        express = rows if express == 0 else express

        express_matrix = self.express_matrix_from_rows_and_express(
            rows=rows, express=express
        )

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
            if type(traps) is TrapNonUniformHeightDistribution:
                trap_managers.append(
                    TrapManagerNonUniformHeightDistribution(traps=trap_group, rows=rows)
                )
            else:
                trap_managers.append(TrapManager(traps=trap_group, rows=rows))

        # Run each independent column of pixels
        for column_index in range(columns):

            # Each calculation of the effects of traps on the pixels
            for express_index in range(express):

                # Reset the trap states
                for trap_manager in trap_managers:
                    trap_manager.reset_traps_for_next_express_loop()

                # Each pixel
                for row_index in range(rows):
                    phase = row_index % phases
                    express_multiplier = express_matrix[express_index, row_index]
                    if express_multiplier == 0:
                        continue

                    # Initial number of electrons available for trapping
                    electrons_initial = image[row_index, column_index]
                    electrons_available = electrons_initial

                    # Release
                    electrons_released = 0
                    for trap_manager in trap_managers:
                        electrons_released += trap_manager.electrons_released_in_pixel(
                            time=clocker.sequence[phase],
                            width=ccd_volume.phase_widths[phase],
                        )
                    electrons_available += electrons_released

                    # Capture
                    electrons_captured = 0
                    for trap_manager in trap_managers:
                        electrons_captured += trap_manager.electrons_captured_in_pixel(
                            electrons_available=electrons_available,
                            ccd_volume=ccd_volume.extract_phase(phase),
                            width=ccd_volume.phase_widths[phase],
                        )

                    # Total change to electrons in pixel
                    electrons_initial += (
                        electrons_released - electrons_captured
                    ) * express_multiplier

                    image[row_index, column_index] = electrons_initial

        # Recombine the image for multi-phase clocking
        if phases > 1:
            image = image.reshape((int(rows / phases), phases, columns)).sum(axis=1)

        return image

    def add_cti(
        self,
        image,
        parallel_clocker=None,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_clocker=None,
        serial_traps=None,
        serial_ccd_volume=None,
    ):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons 
        along their independent columns, for parallel and/or serial clocking.

        Parameters
        ----------
        image : np.ndarray
            The input array of pixel values.
        parallel_clocker : Clocker
            The object describing the clocking read-out electronics for parallel 
            clocking.
        parallel_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for parallel clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        parallel_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for parallel clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as parallel_clocker.sequence.
        serial_clocker : Clocker
            The object describing the clocking read-out electronics for serial 
            clocking.
        serial_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for serial clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        serial_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for serial clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as serial_clocker.sequence.

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

            image = self._add_cti_to_image(
                image=image,
                clocker=parallel_clocker,
                traps=parallel_traps,
                ccd_volume=parallel_ccd_volume,
                express=self.parallel_express,
            )

        if serial_traps is not None:

            image = image.T.copy()

            image = self._add_cti_to_image(
                image=image,
                clocker=serial_clocker,
                traps=serial_traps,
                ccd_volume=serial_ccd_volume,
                express=self.serial_express,
            )

            image = image.T

        return image

    def remove_cti(
        self,
        image,
        parallel_clocker=None,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_clocker=None,
        serial_traps=None,
        serial_ccd_volume=None,
    ):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons 
        along their independent columns, for parallel and/or serial clocking.

        Parameters
        ----------
        image : np.ndarray
            The input array of pixel values.
        parallel_clocker : Clocker
            The object describing the clocking read-out electronics for parallel 
            clocking.
        parallel_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for parallel clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        parallel_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for parallel clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as parallel_clocker.sequence.
        serial_clocker : Clocker
            The object describing the clocking read-out electronics for serial 
            clocking.
        serial_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for serial clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        serial_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for serial clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as serial_clocker.sequence.

        Returns
        ---------
        image : np.ndarray
            The output array of pixel values with CTI removed.
        """

        # Initialise the iterative estimate of removed CTI
        image_remove_cti = deepcopy(image)

        # Estimate the image with removed CTI more precisely each iteration
        for iteration in range(self.iterations):

            image_add_cti = self.add_cti(
                image=image_remove_cti,
                parallel_clocker=parallel_clocker,
                parallel_traps=parallel_traps,
                parallel_ccd_volume=parallel_ccd_volume,
                serial_clocker=serial_clocker,
                serial_traps=serial_traps,
                serial_ccd_volume=serial_ccd_volume,
            )

            # Improved estimate of removed CTI
            image_remove_cti += image - image_add_cti

        return image_remove_cti
