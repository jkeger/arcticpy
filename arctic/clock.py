""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy


class Clocker(object):
    def __init__(
        self,
        sequence=1,
        iterations=1,
        express=5,
        charge_injection_mode=False,
        readout_offset=0,
    ):
        """
        The CTI Clock for arctic clocking.

        Parameters
        --------
        sequence [float]
            The array or single value of the time between clock shifts.
        iterations : int
            If CTI is being corrected, iterations determines the number of times clocking is run to perform the \
            correction via forward modeling. For adding CTI only one run is required and iterations is ignored.
        express : int
            The factor by which pixel-to-pixel transfers are combined for efficiency.
        charge_injection_mode : bool
            If True, clocking is performed in charge injection line mode, where each pixel is clocked and therefore \
             trailed by traps over the entire CCD (as opposed to its distance from the CCD register).
        readout_offset : int
            Introduces an offset which increases the number of transfers each pixel takes in the parallel direction.
        """
        self.sequence = sequence
        self.iterations = iterations
        self.express = express
        self.charge_injection_mode = charge_injection_mode
        self.readout_offset = readout_offset

    def express_matrix_from_rows(self, rows):
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

        express = rows if self.express == 0 else self.express

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

    def initial_watermarks_from_rows_and_total_traps(self, rows, total_traps):
        """ Initialise the watermark array of trap states.

        Parameters
        -----------
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
        total_traps : int
            The number of trap trap being modelled.

        Returns
        --------
        watermarks : np.ndarray
            Array of watermark heights and fill fractions to describe the trap states. Lists each (active) watermark
            height_e fraction and the corresponding fill fractions of each trap trap. Inactive elements are set to 0.

            [[height_e, fill, fill, ...],
             [height_e, fill, fill, ...],
             ...                       ]
        """
        return np.zeros((rows, 1 + total_traps), dtype=float)

    def add_cti(self, image, traps, ccd_volume):
        """
        Add CTI trails to an image by trapping, releasing, and moving electrons along their independent columns.

        Parameters
        -----------
        image : np.ndarray
            The input array of pixel values.
        traps : [Trap]
            A list of one or more trap objects.
        ccd_volume : CCDVolume
            The object describing the CCD volume.

        Returns
        --------
        image : np.ndarray
            The output array of pixel values.
        """

        # Don't modify the external array passed to this function
        image = deepcopy(image)

        rows, columns = image.shape

        # Default to no express and calculate every step
        express = rows if self.express == 0 else self.express

        express_matrix = self.express_matrix_from_rows(rows=rows)

        # Set up the trap watermarks
        watermarks = self.initial_watermarks_from_rows_and_total_traps(
            rows=rows, total_traps=len(traps)
        )

        # Each independent column of pixels
        for column_index in range(columns):

            # Each calculation of the effects of traps on the pixels
            for express_index in range(express):

                # Reset the trap watermarks
                watermarks.fill(0)

                # Each pixel
                for row_index in range(rows):
                    express_multiplier = express_matrix[
                        express_index, row_index
                    ]
                    if express_multiplier == 0:
                        continue

                    # Initial number of electrons available for trapping
                    electrons_initial = image[row_index, column_index]
                    electrons_available = electrons_initial

                    # Release
                    (
                        electrons_released,
                        watermarks,
                    ) = electrons_released_in_pixel(
                        watermarks=watermarks, traps=traps
                    )
                    electrons_available += electrons_released

                    # Capture
                    (
                        electrons_captured,
                        watermarks,
                    ) = electrons_captured_in_pixel(
                        electrons_available=electrons_available,
                        watermarks=watermarks,
                        traps=traps,
                        ccd_volume=ccd_volume,
                    )

                    # Total change to electrons in pixel
                    electrons_initial += (
                        electrons_released - electrons_captured
                    ) * express_multiplier

                    image[row_index, column_index] = electrons_initial

        return image

    def remove_cti(self, image, traps, ccd_volume):
        """
        Remove CTI trails from an image by modelling the effect of adding CTI.

        Parameters
        -----------
        image : np.ndarray
            The input array of pixel values.
        traps : [Trap]
            A list of one or more trap objects.
        ccd_volume : CCDVolume
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
                image_remove_cti, traps=traps, ccd_volume=ccd_volume
            )

            # Improved estimate of removed CTI
            image_remove_cti += image - image_add_cti

        return image_remove_cti


def electrons_released_in_pixel(watermarks, traps):
    """ Release electrons from traps and update the trap watermarks.

    Parameters
    -----------
    watermarks : np.ndarray
        The initial watermarks. See initial_watermarks_from_rows_and_trap().

    traps : [Species]
        An array of one or more objects describing a trap of trap.

    Returns
    ----------
    electrons_released : float
        The number of released electrons.

    watermarks : np.ndarray
        The updated watermarks. See initial_watermarks_from_rows_and_trap().

    """
    # Initialise the number of released electrons
    electrons_released = 0

    # Find the highest active watermark
    max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1

    # For each watermark
    for watermark_index in range(max_watermark_index + 1):
        # Initialise the number of released electrons from this watermark level
        electrons_released_watermark = 0

        # For each trap trap
        for trap_index, trap in enumerate(traps):
            # Number of released electrons (not yet including the trap density)
            electrons_released_from_trap = trap.electrons_released_from_electrons(
                electrons=watermarks[watermark_index, 1 + trap_index]
            )

            # Update the watermark fill fraction
            watermarks[
                watermark_index, 1 + trap_index
            ] -= electrons_released_from_trap

            # Update the actual number of released electrons
            electrons_released_watermark += (
                electrons_released_from_trap * trap.density
            )

        # Multiply the summed fill fractions by the height
        electrons_released += (
            electrons_released_watermark * watermarks[watermark_index, 0]
        )

    return electrons_released, watermarks


def electrons_captured_by_traps(electron_fractional_height, watermarks, traps):
    """

    Find the total number of electrons that the traps can capture.

    Parameters
    ------------
    electron_fractional_height : float
        The fractional height of the electron cloud in the pixel.

    watermarks : np.ndarray
        The initial watermarks. See initial_watermarks_from_rows_and_trap().

    traps : [TrapSpecies]
        An array of one or more objects describing a trap of trap.

    Returns
    ------------
    electrons_captured : float
        The number of captured electrons.
    """
    # Initialise the number of captured electrons
    electrons_captured = 0

    # The number of traps of each trap
    densities = [trap_trap.density for trap_trap in traps]

    # Find the highest active watermark
    max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1

    # Initialise cumulative watermark height
    cumulative_watermark_height = 0

    # Capture electrons above each existing watermark level below the highest
    for watermark_index in range(max_watermark_index + 1):
        # Update cumulative watermark height
        watermark_height = watermarks[watermark_index, 0]
        cumulative_watermark_height += watermark_height

        # Capture electrons all the way to this watermark (for all trap trap)
        if cumulative_watermark_height < electron_fractional_height:
            electrons_captured += watermark_height * np.sum(
                (1 - watermarks[watermark_index, 1:]) * densities
            )

        # Capture electrons part-way between the previous and this watermark
        else:
            electrons_captured += (
                electron_fractional_height
                - (cumulative_watermark_height - watermark_height)
            ) * np.sum((1 - watermarks[watermark_index, 1:]) * densities)

            # No point in checking even higher watermarks
            return electrons_captured

    # Capture any electrons above the highest existing watermark
    if watermarks[max_watermark_index, 0] < electron_fractional_height:
        electrons_captured += (
            electron_fractional_height - cumulative_watermark_height
        ) * np.sum(densities)

    return electrons_captured


def update_watermarks(electron_fractional_height, watermarks):
    """ Update the trap watermarks for capturing electrons.

    Parameters
    -----------
    electron_fractional_height : float
        The fractional height of the electron cloud in the pixel.

    watermarks : np.ndarray
        The initial watermarks. See initial_watermarks_from_rows_and_trap().

    Returns
    ----------
    watermarks : np.ndarray
        The updated watermarks. See initial_watermarks_from_rows_and_trap().
    """

    # Find the highest active watermark
    max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1
    if max_watermark_index == -1:
        max_watermark_index = 0

    # Cumulative watermark heights
    cumulative_watermark_height = np.cumsum(
        watermarks[: max_watermark_index + 1, 0]
    )

    # If all watermarks will be overwritten
    if cumulative_watermark_height[-1] < electron_fractional_height:
        # Overwrite the first watermark
        watermarks[0, 0] = electron_fractional_height
        watermarks[0, 1:] = 1

        # Remove all higher watermarks
        watermarks[1:, :] = 0

        return watermarks

    # Find the first watermark above the cloud, which won't be fully overwritten
    watermark_index_above_cloud = np.argmax(
        electron_fractional_height < cumulative_watermark_height
    )

    # If some will be overwritten
    if 0 < watermark_index_above_cloud:
        # Edit the partially overwritten watermark
        watermarks[watermark_index_above_cloud, 0] -= (
            electron_fractional_height
            - cumulative_watermark_height[watermark_index_above_cloud - 1]
        )

        # Remove the no-longer-needed overwritten watermarks
        watermarks[: watermark_index_above_cloud - 1, :] = 0

        # Move the no-longer-needed watermarks to the end of the list
        watermarks = np.roll(
            watermarks, 1 - watermark_index_above_cloud, axis=0
        )

        # Edit the new first watermark
        watermarks[0, 0] = electron_fractional_height
        watermarks[0, 1:] = 1

    # If none will be overwritten
    else:
        # Move an empty watermark to the start of the list
        watermarks = np.roll(watermarks, 1, axis=0)

        # Edit the partially overwritten watermark
        watermarks[1, 0] -= electron_fractional_height

        # Edit the new first watermark
        watermarks[0, 0] = electron_fractional_height
        watermarks[0, 1:] = 1

    return watermarks


def update_watermarks_not_enough(
    electron_fractional_height, watermarks, enough
):
    """
    Update the trap watermarks for capturing electrons when not enough are available to fill every trap below the
    cloud height (rare!).

    Like update_trap_wmk_capture(), but instead of setting filled trap fractions to 1, increase them by the enough
    fraction towards 1.

    Parameters
    ----------
    electron_fractional_height : float
        The fractional height of the electron cloud in the pixel.
    watermarks : np.ndarray
        The initial watermarks. See initial_watermarks_from_rows_and_trap().
    enough : float
        The ratio of available electrons to traps up to this height.

    Returns
    --------
    watermarks : np.ndarray
        The updated watermarks. See initial_watermarks_from_rows_and_trap().
    """
    # Find the highest active watermark
    max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1
    if max_watermark_index == -1:
        max_watermark_index = 0

    # If the first capture
    if max_watermark_index == 0:
        # Edit the new watermark
        watermarks[0, 0] = electron_fractional_height
        watermarks[0, 1:] = enough

        return watermarks

    # Cumulative watermark heights
    cumulative_watermark_height = np.cumsum(
        watermarks[: max_watermark_index + 1, 0]
    )

    # Find the first watermark above the cloud, which won't be fully overwritten
    watermark_index_above_height = np.argmax(
        electron_fractional_height < cumulative_watermark_height
    )

    # If all watermarks will be overwritten
    if cumulative_watermark_height[-1] < electron_fractional_height:
        # Do the same as the only-some case (unlike update_trap_wmk_capture())
        watermark_index_above_height = max_watermark_index

    # If some (or all) will be overwritten
    if 0 < watermark_index_above_height:
        # Move one new empty watermark to the start of the list
        watermarks = np.roll(watermarks, 1, axis=0)

        # Reorder the relevant watermarks near the start of the list
        watermarks[: 2 * watermark_index_above_height] = watermarks[
            1 : 2 * watermark_index_above_height + 1
        ]

        # Edit the new watermarks' fill fractions to the original fill plus (1 -
        # original fill) * enough.
        # e.g. enough = 0.5 --> fill half way to 1.
        watermarks[: watermark_index_above_height + 1, 1:] = (
            watermarks[: watermark_index_above_height + 1, 1:] * (1 - enough)
            + enough
        )

        # If all watermarks will be overwritten
        if cumulative_watermark_height[-1] < electron_fractional_height:
            # Edit the new highest watermark
            watermarks[watermark_index_above_height + 1, 0] = (
                electron_fractional_height - cumulative_watermark_height[-1]
            )
            watermarks[watermark_index_above_height + 1, 1:] = enough
        else:
            # Edit the new watermarks' heights
            watermarks[
                watermark_index_above_height : watermark_index_above_height + 2,
                0,
            ] *= (1 - enough)

    # If none will be overwritten
    else:
        # Move an empty watermark to the start of the list
        watermarks = np.roll(watermarks, 1, axis=0)

        # Edit the partially overwritten watermark
        watermarks[1, 0] -= electron_fractional_height

        # Edit the new first watermark
        watermarks[0, 0] = electron_fractional_height
        watermarks[0, 1:] = watermarks[1, 1:] * (1 - enough) + enough

    return watermarks


def electrons_captured_in_pixel(
    electrons_available, watermarks, traps, ccd_volume
):
    """
    Capture electrons in traps.

    Parameters
    -----------
    electrons_available : float
        The number of available electrons for trapping.
    watermarks : np.ndarray
        The initial watermarks. See initial_watermarks_from_rows_and_trap().
    traps : [Trap]
        An array of one or more objects describing a trap of trap.
    ccd_volume : CCDVolume
        The object describing the CCD.

    Returns
    ---------
    electrons_captured : float
        The number of captured electrons.
    watermarks : np.ndarray
        The updated watermarks. See initial_watermarks_from_rows_and_trap().
    """

    # Zero capture if no electrons are high enough to be trapped
    if electrons_available < ccd_volume.well_notch_depth:
        return 0, watermarks

    # The fractional height the electron cloud reaches in the pixel well
    electron_fractional_height = ccd_volume.electron_fractional_height_from_electrons(
        electrons=electrons_available
    )

    # Find the number of electrons that should be captured
    electrons_captured = electrons_captured_by_traps(
        electron_fractional_height=electron_fractional_height,
        watermarks=watermarks,
        traps=traps,
    )

    # Stop if no capture
    if electrons_captured == 0:
        return electrons_captured, watermarks

    # Check whether enough electrons are available to be captured
    enough = electrons_available / electrons_captured

    # Update watermark levels
    if 1 < enough:
        watermarks = update_watermarks(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
        )
    else:
        watermarks = update_watermarks_not_enough(
            electron_fractional_height=electron_fractional_height,
            watermarks=watermarks,
            enough=enough,
        )
        # Reduce the final number of captured electrons
        electrons_captured *= enough

    return electrons_captured, watermarks
