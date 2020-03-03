""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy

from arctic.traps import (
    TrapNonUniformDistribution,
    TrapManager,
    TrapManagerNonUniformDistribution,
)


class Clocker(object):
    def __init__(
        self,
        iterations=1,
        parallel_sequence=[1],
        parallel_phase_widths=[1],
        parallel_express=0,
        parallel_charge_injection_mode=False,
        parallel_readout_offset=0,
        parallel_initial_phase=0,
        serial_sequence=[1],
        serial_phase_widths=[1],
        serial_express=0,
        serial_readout_offset=0,
        serial_initial_phase=0,
    ):
        """
        The parameters for the clocking of electrons across the pixels.

        Parameters
        ----------
        iterations : int
            If CTI is being corrected, iterations determines the number of times 
            clocking is run to perform the correction via forward modeling. For 
            adding CTI only one run is required and iterations is ignored.
        parallel_sequence : float or [float]
            The array or single value of the time between clock shifts in each 
            phase or single phase for parallel clocking. The trap release 
            lifetimes must be in the same units. Different CCDVolume parameters 
            may be used for each phase.
        parallel_phase_widths : float or [float]
            The array or single value of the physical width of each phase as a 
            fraction of the pixel for parallel multi-phase clocking. Must be the 
            same size as parallel_sequence.
        parallel_express : int
            The factor by which pixel-to-pixel transfers are combined for 
            efficiency for parallel clocking.
        parallel_charge_injection_mode : bool
            If True, parallel clocking is performed in charge injection line 
            mode, where each pixel is clocked and therefore trailed by traps 
            over the entire CCD (as opposed to its distance from the CCD register).
        parallel_readout_offset : int
            Introduces an offset which increases the number of transfers each 
            pixel takes in the parallel direction.
        parallel_initial_phase : int
            For multi-phase parallel clocking, the initial phase in which the 
            electrons start when the input image is divided into the separate 
            phases.
        serial_sequence : float or [float]
            The array or single value of the time between clock shifts in each 
            phase or single phase for serial clocking. The trap release 
            lifetimes must be in the same units. Different CCDVolume parameters 
            may be used for each phase.
        serial_phase_widths : float or [float]
            The array or single value of the physical width of each phase as a 
            fraction of the pixel for serial multi-phase clocking. Must be the 
            same size as serial_sequence.
        serial_express : int
            The factor by which pixel-to-pixel transfers are combined for 
            efficiency for serial clocking.
        serial_readout_offset : int
            Introduces an offset which increases the number of transfers each 
            pixel takes in the serial direction.
        serial_initial_phase : int
            For multi-phase serial clocking, the initial phase in which the 
            electrons start when the input image is divided into the separate 
            phases.
        """

        self.iterations = iterations

        # Make sure the arrays are arrays
        if not isinstance(parallel_sequence, list):
            parallel_sequence = [parallel_sequence]
        if not isinstance(parallel_phase_widths, list):
            parallel_phase_widths = [parallel_phase_widths]
        if not isinstance(serial_sequence, list):
            serial_sequence = [serial_sequence]
        if not isinstance(serial_phase_widths, list):
            serial_phase_widths = [serial_phase_widths]

        self.parallel_sequence = parallel_sequence
        self.parallel_phase_widths = parallel_phase_widths
        self.parallel_express = parallel_express
        self.parallel_charge_injection_mode = parallel_charge_injection_mode
        self.parallel_readout_offset = parallel_readout_offset
        self.parallel_initial_phase = parallel_initial_phase

        self.serial_sequence = serial_sequence
        self.serial_phase_widths = serial_phase_widths
        self.serial_express = serial_express
        self.serial_readout_offset = serial_readout_offset
        self.serial_initial_phase = serial_initial_phase

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

    def _add_cti_to_image(
        self, image, sequence, phase_widths, traps, ccd_volume, express, 
        initial_phase
    ):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons 
        along their independent columns.

        Parameters
        ----------
        image : np.ndarray
            The input array of pixel values.
        sequence : float or [float]
            The array or single value of the time between clock shifts in each 
            phase or single phase. The trap release lifetimes must be in the 
            same units. Different CCDVolume parameters may be used for each 
            phase.
        phase_widths : float or [float]
            The array or single value of the physical width of each phase as a 
            fraction of the pixel for multi-phase clocking. Must be the same
            size as sequence.
        traps : [Trap] or [[Trap]]
            A list of one or more trap objects. To use different types of traps 
            that will require different watermark levels, pass a 2D list of 
            lists, i.e. a list containing lists of one or more traps for each 
            type. 
        ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume. For multi-phase clocking 
            optionally enter use a list of different CCD volumes for each phase, 
            in the same size list as the clock sequence.
        express : int
            The factor by which pixel-to-pixel transfers are combined for 
            efficiency.
        initial_phase : int
            For multi-phase clocking, the initial phase in which the electrons 
            start when the input image is divided into the separate phases.

        Returns
        -------
        image : np.ndarray
            The output array of pixel values.
        """

        rows, columns = image.shape

        # Make sure the arrays are arrays
        if not isinstance(sequence, list):
            sequence = [sequence]
        if not isinstance(phase_widths, list):
            phase_widths = [phase_widths]
        if not isinstance(traps[0], list):
            traps = [traps]
        if not isinstance(ccd_volume, list):
            ccd_volume = [ccd_volume]

        phases = len(sequence)
        assert len(phase_widths) == phases
        assert np.amax(phase_widths) <= 1
        
        # Assume the same CCD volume for all phases if not otherwise specified
        if len(ccd_volume) == 1 and len(ccd_volume) != phases:
            ccd_volume *= phases
        assert len(ccd_volume) == phases

        # Default to no express and calculate every step
        express = rows if express == 0 else express

        express_matrix = self.express_matrix_from_rows_and_express(
            rows=rows, express=express
        )

        # Prepare the image and express for multi-phase clocking
        if phases != 1:
            new_image = np.zeros((rows * phases, columns))
            new_image[initial_phase::phases] = image
            image = new_image
            rows, columns = image.shape
            
            express_matrix = np.repeat(express_matrix, phases, axis=1)

        # Set up the array of trap managers
        trap_managers = []
        for trap_group in traps:
            if type(traps) is TrapNonUniformDistribution:
                trap_managers.append(
                    TrapManagerNonUniformDistribution(
                        traps=trap_group, rows=rows
                    )
                )
            else:
                trap_managers.append(TrapManager(traps=trap_group, rows=rows))

        # Clock each independent column of pixels
        for column_index in range(columns):

            # Each calculation of the effects of traps on the pixels
            for express_index in range(express):

                # Reset the trap states
                for trap_manager in trap_managers:
                    trap_manager.reset_traps_for_next_express_loop()

                # Each pixel
                for row_index in range(rows):
                    phase = row_index % phases
                    express_multiplier = express_matrix[
                        express_index, row_index
                    ]
                    if express_multiplier == 0:
                        continue

                    # Initial number of electrons available for trapping
                    electrons_initial = image[row_index, column_index]
                    electrons_available = electrons_initial

                    # Release
                    electrons_released = 0
                    for trap_manager in trap_managers:
                        electrons_released += trap_manager.electrons_released_in_pixel(
                            time=sequence[phase], 
                            width=phase_widths[phase]
                        )
                    electrons_available += electrons_released

                    # Capture
                    electrons_captured = 0
                    for trap_manager in trap_managers:
                        electrons_captured += trap_manager.electrons_captured_in_pixel(
                            electrons_available=electrons_available,
                            ccd_volume=ccd_volume[phase], 
                            width=phase_widths[phase],
                        )

                    # Total change to electrons in pixel
                    electrons_initial += (
                        electrons_released - electrons_captured
                    ) * express_multiplier

                    image[row_index, column_index] = electrons_initial

        # Recombine the image for multi-phase clocking
        if phases != 1:
            image = image.reshape((int(rows / phases), phases, columns)).sum(
                axis=1
            )

        return image

    def add_cti(
        self,
        image,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
    ):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons 
        along their independent columns, for parallel and serial clocking.

        Parameters
        ----------
        image : np.ndarray
            The input array of pixel values.
        parallel_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for parallel clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        parallel_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for parallel clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as parallel_sequence.
        serial_traps : [Trap] or [[Trap]]
            A list of one or more trap objects for serial clocking. To use 
            different types of traps that will require different watermark 
            levels, pass a 2D list of lists, i.e. a list containing lists of 
            one or more traps for each type. 
        serial_ccd_volume : CCDVolume or [CCDVolume]
            The object describing the CCD volume for serial clocking. For 
            multi-phase clocking optionally use a list of different CCD volumes
            for each phase, in the same size list as serial_sequence.

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
                sequence=self.parallel_sequence,
                phase_widths=self.parallel_phase_widths,
                traps=parallel_traps,
                ccd_volume=parallel_ccd_volume,
                express=self.parallel_express,
                initial_phase=self.parallel_initial_phase,
            )

        if serial_traps is not None:

            image = image.T.copy()

            image = self._add_cti_to_image(
                image=image,
                sequence=self.serial_sequence,
                phase_widths=self.serial_phase_widths,
                traps=serial_traps,
                ccd_volume=serial_ccd_volume,
                express=self.serial_express,
                initial_phase=self.serial_initial_phase,
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
