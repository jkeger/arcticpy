""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy

from arctic.traps import TrapManager


class Clocker(object):
    def __init__(
        self,
        iterations=1,
        parallel_sequence=1,
        parallel_express=5,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
        serial_sequence=1,
        serial_express=5,
        serial_readout_offset=0,
    ):
        """
        The CTI Clock for arctic clocking.

        Parameters
        ----------
        parallel_sequence : [float]
            The array or single value of the time between clock shifts.
        iterations : int
            If CTI is being corrected, iterations determines the number of times clocking is run to perform the \
            correction via forward modeling. For adding CTI only one run is required and iterations is ignored.
        parallel_express : int
            The factor by which pixel-to-pixel transfers are combined for efficiency.
        parallel_charge_injection_mode : bool
            If True, clocking is performed in charge injection line mode, where each pixel is clocked and therefore \
             trailed by traps over the entire CCD (as opposed to its distance from the CCD register).
        parallel_readout_offset : int
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """

        self.iterations = iterations

        self.parallel_sequence = parallel_sequence
        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.parallel_readout_offset = parallel_readout_offset

        self.serial_sequence = serial_sequence
        self.serial_express = serial_express
        self.serial_readout_offset = serial_readout_offset

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

    def add_cti(
        self,
        image,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
    ):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons along their independent columns.

        Parameters
        ----------
        image : np.ndarray
            The input array of pixel values.
        parallel_traps : [Trap]
            A list of one or more trap objects.
        parallel_ccd_volume : CCDVolume
            The object describing the CCD volume.

        Returns
        -------
        image : np.ndarray
            The output array of pixel values.
        """

        # Don't modify the external array passed to this function
        image = deepcopy(image)

        if parallel_traps is not None:

            image = self._add_cti_to_image(
                image=image,
                traps=parallel_traps,
                ccd_volume=parallel_ccd_volume,
                express=self.parallel_express,
            )

        if serial_traps is not None:

            image = image.T.copy()

            image = self._add_cti_to_image(
                image=image,
                traps=serial_traps,
                ccd_volume=serial_ccd_volume,
                express=self.serial_express,
            )

            image = image.T

        return image

    def remove_cti(
        self,
        image,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
    ):
        """
        Remove CTI trails from an image by modelling the effect of adding CTI.

        Parameters
        -----------
        image : np.ndarray
            The input array of pixel values.
        parallel_traps : [Trap]
            A list of one or more trap objects.
        parallel_ccd_volume : CCDVolume
            The object describing the CCD volume.

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
                parallel_traps=parallel_traps,
                parallel_ccd_volume=parallel_ccd_volume,
                serial_traps=serial_traps,
                serial_ccd_volume=serial_ccd_volume,
            )

            # Improved estimate of removed CTI
            image_remove_cti += image - image_add_cti

        return image_remove_cti

    def _add_cti_to_image(self, image, traps, ccd_volume, express):

        rows, columns = image.shape

        # Default to no express and calculate every step
        express = rows if express == 0 else express

        express_matrix = self.express_matrix_from_rows_and_express(
            rows=rows, express=express
        )

        # Set up the trap manager
        trap_manager = TrapManager(traps=traps, rows=rows)

        # Each independent column of pixels
        for column_index in range(columns):

            # Each calculation of the effects of traps on the pixels
            for express_index in range(express):

                # Reset the trap states
                trap_manager.reset_traps_for_next_express_loop()

                # Each pixel
                for row_index in range(rows):
                    express_multiplier = express_matrix[express_index, row_index]
                    if express_multiplier == 0:
                        continue

                    # Initial number of electrons available for trapping
                    electrons_initial = image[row_index, column_index]
                    electrons_available = electrons_initial

                    # Release
                    electrons_released = trap_manager.electrons_released_in_pixel()
                    electrons_available += electrons_released

                    # Capture
                    electrons_captured = trap_manager.electrons_captured_in_pixel(
                        electrons_available=electrons_available, ccd_volume=ccd_volume
                    )

                    # Total change to electrons in pixel
                    electrons_initial += (
                        electrons_released - electrons_captured
                    ) * express_multiplier

                    image[row_index, column_index] = electrons_initial

        return image

