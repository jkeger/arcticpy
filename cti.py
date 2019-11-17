""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk
    
    WIP...
"""
import numpy as np


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
def init_A2_trap_wmk_height_fill(num_column, num_species):
    """ Initialise the watermark array of trap states.
    
        Args:
            num_column (int)
                The number of columns in the image. i.e. the maximum number of 
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
    return np.zeros((num_column, 1 + num_species), dtype=float)


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

    # Find the first watermark above the cloud that won't be fully overwritten
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
