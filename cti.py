""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk
    
    WIP...
"""
import numpy as np
from copy import deepcopy


# //////////////////////////////////////////////////////////////////////////// #
#                               Utility Functions                              #
# //////////////////////////////////////////////////////////////////////////// #
def set_min_max(value, min, max):
    """ Fix a value between a minimum and maximum. """
    if value < min:
        return min
    elif max < value:
        return max
    else:
        return value


def format_array_string(array, format): ### just for debugging
    """ Return a print-ready string of an array's contents in a given format.

        Args:
            array ([])
                An array of values that can be printed with the given format.

            format (str)
                A printing format, e.g. "%d", "%.6g".
                
                Custom options:
                    "string":   Include quotation marks around each string.
                    "dorf":     Int or float if decimal places are needed.

        Returns:
            string (str)
                The formatted string.
    """
    string = ""

    # Append each element
    # 1D
    if len(np.shape(array)) == 1:
        for x in array:
            if x is None:
                string += "None, "
            else:
                # Custom formats
                if format == "string":
                    string += '"%s", ' % x
                elif format == "dorf":
                    string += "{0}, ".format(
                        str(round(x, 1) if x % 1 else int(x))
                    )
                # Standard formats
                else:
                    string += "%s, " % (format % x)
    # Recursive for higher dimensions
    else:
        for arr in array:
            string += "%s, \n " % format_array_string(arr, format)

    # Add brackets and remove the trailing comma
    return "[%s]" % string[:-2]


# //////////////////////////////////////////////////////////////////////////// #
#                               Classes                                        #
# //////////////////////////////////////////////////////////////////////////// #
class CCD(object):
    def __init__(
        self,
        well_fill_alpha,
        well_fill_beta,
        well_fill_gamma,
        well_max_height,
        well_notch_height,
    ):
        """ Properties of the CCD.
        
            Args:
                well_fill_alpha : float
                    The volume-filling coefficient, alpha, of how an electron 
                    cloud fills the volume of a pixel.
                
                well_fill_beta : float
                    The volume-filling power, beta, of how an electron cloud 
                    fills the volume of a pixel.
                
                well_fill_gamma : float
                    The volume-filling constant, gamma, of how an electron cloud 
                    fills the volume of a pixel.
                
                well_max_height : float
                    The total depth of the CCD well, i.e. the maximum number of 
                    electrons per pixel.
                
                well_notch_height : float
                    The depth of the CCD notch, i.e. the minimum number of 
                    electrons per pixel that are relevant for trapping.
        """
        self.well_fill_alpha = well_fill_alpha
        self.well_fill_beta = well_fill_beta
        self.well_fill_gamma = well_fill_gamma
        self.well_max_height = well_max_height
        self.well_notch_height = well_notch_height
        self.well_range = self.well_max_height - self.well_notch_height

    def height_e(self, e_avai):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """
        height_e = (
            (e_avai - self.well_notch_height) / self.well_range
        ) ** self.well_fill_beta

        return set_min_max(height_e, 0, 1)


class Clock(object):
    def __init__(self, A1_sequence):
        """ The clock that controls the transfer of electrons.

            Args:
                A1_sequence ([float])
                    The array or single value of the time between clock shifts.
        """
        self.A1_sequence = A1_sequence


class TrapSpecies(object):
    def __init__(self, density, lifetime):
        """ A species of trap.
        
            Args:                
                density : float
                    The number of traps per pixel.
                
                lifetime : float
                    The lifetime for release of an electron from the trap,
                    in clock cycles.
        """
        # Required
        self.density = density
        self.lifetime = lifetime

        # Derived
        self.exponential_factor = 1 - np.exp(-1 / self.lifetime)

    def e_rele(self, e_init):
        """ Calculate the number of released electrons from the trap.
        
            Args:
                e_init : float
                    The initial number of trapped electrons.
                
            Returns:
                e_rele : float
                    The number of released electrons.
        """
        return e_init * self.exponential_factor


# //////////////////////////////////////////////////////////////////////////// #
#                               Support Functions                              #
# //////////////////////////////////////////////////////////////////////////// #
def create_express_multiplier(express, num_row, do_charge_inj=False):
    """ Calculate the values by which the effects of clocking each pixel must be 
        multiplied when using the express algorithm. See Massey et al. (2014), 
        section 2.1.5.
        
        Args:
            express : int
                The express parameter = (maximum number of transfers) / (number 
                of computed tranfers), i.e. the number of computed transfers per 
                pixel. So express = 1 --> maxmimum speed; express = (maximum 
                number of transfers) --> maximum accuracy.
                
                Must be a factor of num_row.
            
            num_row : int
                The number of rows in the CCD and hence the maximum number of 
                transfers.
            
            do_charge_inj : bool
                If True, then assume that all pixels must be transferred along 
                the full number of rows. Default False.
        
        Returns:
            A2_express_multiplier : [[float]]
                The express multiplier values for each pixel.
    """
    assert (
        num_row % express == 0
    ), "express must be a factor of the number of rows"
    express_max = int(num_row / express)
    
    # Simpler version for charge-injection
    if do_charge_inj:
        return np.ones((express, num_row), dtype=int) * express_max

    # Initialise the array
    A2_express_multiplier = np.empty((express, num_row), dtype=int)
    A2_express_multiplier[:] = np.arange(1, num_row + 1)

    # Offset each row to account for the pixels that have already been read out
    for i_exp in range(express):
        A2_express_multiplier[i_exp] -= i_exp * express_max

    # Set all values to between 0 and express_max
    A2_express_multiplier[A2_express_multiplier < 0] = 0
    A2_express_multiplier[express_max < A2_express_multiplier] = express_max

    return A2_express_multiplier


def init_A2_trap_wmk_height_fill(num_row, num_species):
    """ Initialise the watermark array of trap states.
    
        Args:
            num_row (int)
                The number of rows in the image. i.e. the maximum number of 
                possible electron trap/release events.
            
            num_species (int)
                The number of trap species being modelled.
    
        Returns:
            A2_trap_wmk_height_fill : [[float]]
                Array of watermark heights and fill fractions to describe the 
                trap states. Lists each (active) watermark height_e fraction and  
                the corresponding fill fractions of each trap species. Inactive 
                elements are set to 0.
                
                [[height_e, fill, fill, ...], 
                 [height_e, fill, fill, ...],
                 ...                       ]
    """
    return np.zeros((num_row, 1 + num_species), dtype=float)


def release_electrons_in_pixel(A2_trap_wmk_height_fill, A1_trap_species):
    """ Release electrons from traps.
    
        Find the total number of electrons that the traps release and update the
        trap watermarks.
    
        Args:
            A2_trap_wmk_height_fill : [[float]]
                The initial watermarks. See init_A2_trap_wmk_height_fill().
                
            A1_trap_species : [TrapSpecies]
                An array of one or more objects describing a species of trap.
                
        Returns:
            e_rele : float
                The number of released electrons.
                
            A2_trap_wmk_height_fill : [[float]]
                The updated watermarks. See init_A2_trap_wmk_height_fill().
                
        TEST: test_release_electrons_in_pixel()
    """
    # Initialise the number of released electrons
    e_rele = 0

    # The number of traps of each species
    A1_trap_density = [trap_species.density for trap_species in A1_trap_species]

    # Find the highest active watermark
    i_wmk_max = np.argmax(A2_trap_wmk_height_fill[:, 0] == 0) - 1

    # For each watermark
    for i_wmk in range(i_wmk_max + 1):
        # Initialise the number of released electrons from this watermark level
        e_rele_wmk = 0

        # For each trap species
        for i_spec, species in enumerate(A1_trap_species):
            # Number of released electrons (not yet including the trap density)
            e_rele_spec = species.e_rele(
                A2_trap_wmk_height_fill[i_wmk, 1 + i_spec]
            )

            # Update the watermark fill fraction
            A2_trap_wmk_height_fill[i_wmk, 1 + i_spec] -= e_rele_spec

            # Update the actual number of released electrons
            e_rele_wmk += e_rele_spec * A1_trap_density[i_spec]

        # Multiply the summed fill fractions by the height
        e_rele += e_rele_wmk * A2_trap_wmk_height_fill[i_wmk, 0]

    return e_rele, A2_trap_wmk_height_fill


def capture_electrons(height_e, A2_trap_wmk_height_fill, A1_trap_species):
    """ Find the total number of electrons that the traps can capture.
    
        Args:
            height_e : float
                The fractional height of the electron cloud in the pixel.
        
            A2_trap_wmk_height_fill : [[float]]
                The initial watermarks. See init_A2_trap_wmk_height_fill().
                
            A1_trap_species : [TrapSpecies]
                An array of one or more objects describing a species of trap.
            
        Returns:
            e_capt : float
                The number of captured electrons.
        
        TEST: test_capture_electrons()
    """
    # Initialise the number of captured electrons
    e_capt = 0

    # The number of traps of each species
    A1_trap_density = [trap_species.density for trap_species in A1_trap_species]

    # Find the highest active watermark
    i_wmk_max = np.argmax(A2_trap_wmk_height_fill[:, 0] == 0) - 1

    # Initialise cumulative watermark height
    height_wmk_cum = 0

    # Capture electrons above each existing watermark level below the highest
    for i_wmk in range(i_wmk_max + 1):
        # Update cumulative watermark height
        height_wmk = A2_trap_wmk_height_fill[i_wmk, 0]
        height_wmk_cum += height_wmk

        # Capture electrons all the way to this watermark (for all trap species)
        if height_wmk_cum < height_e:
            e_capt += height_wmk * np.sum(
                (1 - A2_trap_wmk_height_fill[i_wmk, 1:]) * A1_trap_density
            )

        # Capture electrons part-way between the previous and this watermark
        else:
            e_capt += (height_e - (height_wmk_cum - height_wmk)) * np.sum(
                (1 - A2_trap_wmk_height_fill[i_wmk, 1:]) * A1_trap_density
            )

            # No point in checking even higher watermarks
            return e_capt

    # Capture any electrons above the highest existing watermark
    if A2_trap_wmk_height_fill[i_wmk_max, 0] < height_e:
        e_capt += (height_e - height_wmk_cum) * np.sum(A1_trap_density)

    return e_capt


def update_trap_wmk_capture(height_e, A2_trap_wmk_height_fill):
    """ Update the trap watermarks for capturing electrons.
    
        Args:
            height_e : float
                The fractional height of the electron cloud in the pixel.
        
            A2_trap_wmk_height_fill : [[float]]
                The initial watermarks. See init_A2_trap_wmk_height_fill().
        
        Returns:
            A2_trap_wmk_height_fill : [[float]]
                The updated watermarks. See init_A2_trap_wmk_height_fill().
    
        TEST: test_update_trap_wmk_capture()
    """
    # Find the highest active watermark
    i_wmk_max = np.argmax(A2_trap_wmk_height_fill[:, 0] == 0) - 1
    if i_wmk_max == -1:
        i_wmk_max = 0

    # Cumulative watermark heights
    A1_trap_cum_height = np.cumsum(A2_trap_wmk_height_fill[: i_wmk_max + 1, 0])

    # If all watermarks will be overwritten
    if A1_trap_cum_height[-1] < height_e:
        # Overwrite the first watermark
        A2_trap_wmk_height_fill[0, 0] = height_e
        A2_trap_wmk_height_fill[0, 1:] = 1

        # Remove all higher watermarks
        A2_trap_wmk_height_fill[1:, :] = 0

        return A2_trap_wmk_height_fill

    # Find the first watermark above the cloud, which won't be fully overwritten
    i_wmk_above = np.argmax(height_e < A1_trap_cum_height)

    # If some will be overwritten
    if 0 < i_wmk_above:
        # Edit the partially overwritten watermark
        A2_trap_wmk_height_fill[i_wmk_above, 0] -= (
            height_e - A1_trap_cum_height[i_wmk_above - 1]
        )

        # Remove the no-longer-needed overwritten watermarks
        A2_trap_wmk_height_fill[: i_wmk_above - 1, :] = 0

        # Move the no-longer-needed watermarks to the end of the list
        A2_trap_wmk_height_fill = np.roll(
            A2_trap_wmk_height_fill, 1 - i_wmk_above, axis=0
        )

        # Edit the new first watermark
        A2_trap_wmk_height_fill[0, 0] = height_e
        A2_trap_wmk_height_fill[0, 1:] = 1

    # If none will be overwritten
    else:
        # Move an empty watermark to the start of the list
        A2_trap_wmk_height_fill = np.roll(A2_trap_wmk_height_fill, 1, axis=0)

        # Edit the partially overwritten watermark
        A2_trap_wmk_height_fill[1, 0] -= height_e

        # Edit the new first watermark
        A2_trap_wmk_height_fill[0, 0] = height_e
        A2_trap_wmk_height_fill[0, 1:] = 1

    return A2_trap_wmk_height_fill


def update_trap_wmk_capture_not_enough(
    height_e, A2_trap_wmk_height_fill, enough
):
    """ Update the trap watermarks for capturing electrons when not enough are 
        available to fill every trap below the cloud height (rare!).
    
        Like update_trap_wmk_capture(), but instead of setting filled trap 
        fractions to 1, increase them by the enough fraction towards 1.
    
        Args:
            height_e : float
                The fractional height of the electron cloud in the pixel.
        
            A2_trap_wmk_height_fill : [[float]]
                The initial watermarks. See init_A2_trap_wmk_height_fill().
        
            enough : float
                The ratio of available electrons to traps up to this height.
        
        Returns:
            A2_trap_wmk_height_fill : [[float]]
                The updated watermarks. See init_A2_trap_wmk_height_fill().
    
        TEST: test_update_trap_wmk_capture_not_enough()
    """
    # Find the highest active watermark
    i_wmk_max = np.argmax(A2_trap_wmk_height_fill[:, 0] == 0) - 1
    if i_wmk_max == -1:
        i_wmk_max = 0

    # If the first capture
    if i_wmk_max == 0:
        # Edit the new watermark
        A2_trap_wmk_height_fill[0, 0] = height_e
        A2_trap_wmk_height_fill[0, 1:] = enough

        return A2_trap_wmk_height_fill

    # Cumulative watermark heights
    A1_trap_cum_height = np.cumsum(A2_trap_wmk_height_fill[: i_wmk_max + 1, 0])

    # Find the first watermark above the cloud, which won't be fully overwritten
    i_wmk_above = np.argmax(height_e < A1_trap_cum_height)

    # If all watermarks will be overwritten
    if A1_trap_cum_height[-1] < height_e:
        # Do the same as the only-some case (unlike update_trap_wmk_capture())
        i_wmk_above = i_wmk_max

    # If some (or all) will be overwritten
    if 0 < i_wmk_above:
        # Move one new empty watermark to the start of the list
        A2_trap_wmk_height_fill = np.roll(A2_trap_wmk_height_fill, 1, axis=0)

        # Reorder the relevant watermarks near the start of the list
        A2_trap_wmk_height_fill[: 2 * i_wmk_above] = A2_trap_wmk_height_fill[
            1 : 2 * i_wmk_above + 1
        ]

        # Edit the new watermarks' fill fractions to the original fill plus (1 -
        # original fill) * enough.
        # e.g. enough = 0.5 --> fill half way to 1.
        A2_trap_wmk_height_fill[: i_wmk_above + 1, 1:] = (
            A2_trap_wmk_height_fill[: i_wmk_above + 1, 1:] * (1 - enough)
            + enough
        )

        # If all watermarks will be overwritten
        if A1_trap_cum_height[-1] < height_e:
            # Edit the new highest watermark
            A2_trap_wmk_height_fill[i_wmk_above + 1, 0] = (
                height_e - A1_trap_cum_height[-1]
            )
            A2_trap_wmk_height_fill[i_wmk_above + 1, 1:] = enough
        else:
            # Edit the new watermarks' heights
            A2_trap_wmk_height_fill[i_wmk_above : i_wmk_above + 2, 0] *= (
                1 - enough
            )

    # If none will be overwritten
    else:
        # Move an empty watermark to the start of the list
        A2_trap_wmk_height_fill = np.roll(A2_trap_wmk_height_fill, 1, axis=0)

        # Edit the partially overwritten watermark
        A2_trap_wmk_height_fill[1, 0] -= height_e

        # Edit the new first watermark
        A2_trap_wmk_height_fill[0, 0] = height_e
        A2_trap_wmk_height_fill[0, 1:] = (
            A2_trap_wmk_height_fill[1, 1:] * (1 - enough) + enough
        )

    return A2_trap_wmk_height_fill


def capture_electrons_in_pixel(
    e_avai, A2_trap_wmk_height_fill, A1_trap_species, ccd
):
    """ Capture electrons in traps.
    
        Args:
            e_avai : float
            The number of available electrons for trapping.
            
            A2_trap_wmk_height_fill : [[float]]
                The initial watermarks. See init_A2_trap_wmk_height_fill().
                
            A1_trap_species : [TrapSpecies]
                An array of one or more objects describing a species of trap.
            
            ccd : CCD
                The object describing the CCD.
            
        Returns:
            e_capt : float
                The number of captured electrons.
                
            A2_trap_wmk_height_fill : [[float]]
                The updated watermarks. See init_A2_trap_wmk_height_fill().
    
        TEST: test_capture_electrons_in_pixel()
    """
    # Initialise the number of captured electrons
    e_capt = 0

    # Zero capture if no electrons are high enough to be trapped
    if e_avai < ccd.well_notch_height:
        return 0, A2_trap_wmk_height_fill

    # The fractional height the electron cloud reaches in the pixel well
    height_e = ccd.height_e(e_avai)

    # Find the number of electrons that should be captured
    e_capt = capture_electrons(
        height_e, A2_trap_wmk_height_fill, A1_trap_species
    )

    # Stop if no capture
    if e_capt == 0:
        return e_capt, A2_trap_wmk_height_fill

    # Check whether enough electrons are available to be captured
    enough = e_avai / e_capt

    # Update watermark levels
    if 1 < enough:
        A2_trap_wmk_height_fill = update_trap_wmk_capture(
            height_e, A2_trap_wmk_height_fill
        )
    else:
        A2_trap_wmk_height_fill = update_trap_wmk_capture_not_enough(
            height_e, A2_trap_wmk_height_fill, enough
        )
        # Reduce the final number of captured electrons
        e_capt *= enough

    return e_capt, A2_trap_wmk_height_fill


# //////////////////////////////////////////////////////////////////////////// #
#                               Primary Functions                              #
# //////////////////////////////////////////////////////////////////////////// #
def add_cti(A2_image, A1_trap_species, ccd, clock, express=0):
    """ Add CTI trails to an image by trapping, releasing, and moving electrons
        along their independent columns.
    
        Args:
            A2_image : [[float]]
                The input array of pixel values.
            
            A1_trap_species : [TrapSpecies]
                An array of one or more objects describing a species of trap.
            
            ccd : CCD
                The object describing the CCD.
            
            clock : Clock
                The object describing the clock sequence.
            
            express : int
                The express level. See create_express_multiplier() and Massey et 
                al. (2014), section 2.1.5. Set = 0 or num_row for no express.
                
        Returns:
            A2_image : [[float]]
                The output array of pixel values.
    """
    # Don't modify the external array passed to this function
    A2_image = deepcopy(A2_image)

    num_row, num_column = A2_image.shape
    num_species = len(A1_trap_species)

    # Default to no express and calculate every step
    if express == 0:
        express = num_row

    A2_express_multiplier = create_express_multiplier(express, num_row)

    # Set up the trap watermarks
    A2_trap_wmk_height_fill = init_A2_trap_wmk_height_fill(num_row, num_species)

    # Each independent column of pixels
    for i_column in range(num_column):

        # Each calculation of the effects of traps on the pixels
        for i_express in range(express):

            # Reset the trap watermarks
            A2_trap_wmk_height_fill.fill(0)

            # Each pixel
            for i_row in range(num_row):
                express_multiplier = A2_express_multiplier[i_express, i_row]
                if express_multiplier == 0:
                    continue

                # Initial number of electrons available for trapping
                e_init = A2_image[i_row, i_column]
                e_avai = e_init

                # Release
                e_rele, A2_trap_wmk_height_fill = release_electrons_in_pixel(
                    A2_trap_wmk_height_fill, A1_trap_species
                )
                e_avai += e_rele

                # Capture
                e_capt, A2_trap_wmk_height_fill = capture_electrons_in_pixel(
                    e_avai, A2_trap_wmk_height_fill, A1_trap_species, ccd
                )

                # Total change to electrons in pixel
                e_init += (e_rele - e_capt) * express_multiplier

                A2_image[i_row, i_column] = e_init

    return A2_image


def remove_cti(iterations, A2_image, A1_trap_species, ccd, clock, express=0):
    """ Remove CTI trails from an image by modelling the effect of adding CTI.
    
        Args:
            iterations : int
                The number of iterations of modelling CTI used to attempt to 
                remove it. Use more iterations to converge on the best recovery
                of the original image, for an increase in computational cost.                
                See Massey et al. (2010), section 3.2.
            
            See add_cti() for the documentation of the other arguments.            
                
        Returns:
            A2_image : [[float]]
                The output array of pixel values with CTI removed.
    """
    # Initialise the iterative estimate of removed CTI
    A2_image_rem = deepcopy(A2_image)

    # Estimate the image with removed CTI more precisely each iteration
    for iter in range(iterations):
        # Add CTI
        A2_image_add = add_cti(
            A2_image_rem, A1_trap_species, ccd, clock, express=express
        )

        # Improved estimate of removed CTI
        A2_image_rem += A2_image - A2_image_add

    return A2_image_rem
