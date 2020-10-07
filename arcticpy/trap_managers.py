import numpy as np
from collections import UserList
from scipy import integrate, optimize
from copy import deepcopy
import warnings
from arcticpy.traps import (
    Trap,
    TrapLifetimeContinuumAbstract,
    TrapLogNormalLifetimeContinuum,
    TrapInstantCapture,
)
from arcticpy.ccd import CCD, CCDPhase

from arcticpy.trap_managers_utils import (
    cy_n_trapped_electrons_from_watermarks,
    cy_update_watermark_volumes_for_cloud_below_highest,
    cy_watermark_index_above_cloud_from_cloud_fractional_volume,
    cy_value_in_cumsum,
)


class AllTrapManager(UserList):
    def __init__(self, traps, max_n_transfers, ccd):
        """
        A list (of a list) of trap managers.
        
        Each trap manager handles a group of trap species that shares watermark 
        levels; these are joined in a list. The list is then repeated for each 
        phase in the CCD pixels (default only 1), as the traps in each phase 
        must be tracked separately. So each trap manager can be accessed as 
        AllTrapManager[trap_group_index][phase].
        
        Parameters
        ----------
        traps : [[Trap]] (or Trap or [Trap])
            A list of one or more trap species. Species listed together in each 
            innermost list must be able to share watermarks - i.e. they are 
            distributed in the same way throughout the pixel volume and their 
            state is stored in the same way by occupancy or time since filling, 
            e.g. 
            [
                [slow_trap, fast_trap], 
                [continuum_lifetime_trap], 
                [surface_trap],
            ]
            
        max_n_transfers : int
            The number of pixels containing traps that charge will be expected 
            to move. This determines the maximum number of possible capture/
            release events that could create new watermark levels, and is used 
            to initialise the watermark array to be only as large as needed, for 
            efficiency.
            
        ccd : CCD
            Configuration of the CCD in which the electrons will move. Used to 
            access the number of phases per pixel, and the fractional volume of 
            a pixel that is filled by a cloud of electrons.
            
        Attributes
        ----------            
        n_electrons_trapped_currently : float
            The number of electrons in traps that are currently being actively 
            monitored.
        
        Methods
        -------
        empty_all_traps()
            Set all trap occupancies to empty.
            
        save()
            Save trap occupancy levels for future reference.
            
        restore()
            Recall trap occupancy levels.
        """

        # Parse inputs
        # If only a single trap species is supplied, still make sure it is a list
        if not isinstance(traps, list):
            traps = [traps]
        if not isinstance(traps[0], list):
            traps = [traps]

        # Replicate trap managers to keep track of traps in different phases separately
        self.data = []
        for phase in range(ccd.n_phases):

            # Set up list of traps in a single phase of the CCD
            trap_managers_this_phase = []
            for trap_group in traps:
                # Use a non-default trap manager if required for the input trap species
                if isinstance(
                    trap_group[0],
                    (TrapLifetimeContinuumAbstract, TrapLogNormalLifetimeContinuum),
                ):
                    trap_manager = TrapManagerTrackTime(
                        traps=trap_group, max_n_transfers=max_n_transfers
                    )
                elif isinstance(trap_group[0], TrapInstantCapture):
                    trap_manager = TrapManagerInstantCapture(
                        traps=trap_group, max_n_transfers=max_n_transfers
                    )
                else:
                    trap_manager = TrapManager(
                        traps=trap_group, max_n_transfers=max_n_transfers
                    )
                trap_manager.n_traps_per_pixel *= ccd.fraction_of_traps_per_phase[phase]
                trap_managers_this_phase.append(trap_manager)

            self.data.append(trap_managers_this_phase)

        # Initialise the empty trap state for future reference
        self._saved_data = None
        self._n_electrons_trapped_in_save = 0.0
        self._n_electrons_trapped_previously = 0.0

    @property
    def n_electrons_trapped_currently(self):
        """ The number of electrons in traps that are currently being actively monitored. """
        n_electrons_trapped_currently = 0

        for trap_manager_phase in self.data:
            for trap_manager in trap_manager_phase:
                n_electrons_trapped_currently += trap_manager.n_trapped_electrons_from_watermarks(
                    trap_manager.watermarks
                )

        return n_electrons_trapped_currently

    def empty_all_traps(self):
        """ Set all trap occupancies to zero """
        for trap_manager_phase in self.data:
            for trap_manager_group in trap_manager_phase:
                trap_manager_group.empty_all_traps()

    def save(self):
        """ Save trap occupancy levels for future reference """
        # This stores far more than necessary. But extracting only the watermark
        # arrays requires overhead.
        self._saved_data = deepcopy(self.data)
        self._n_electrons_trapped_in_save = self.n_electrons_trapped_currently

    def restore(self):
        """ Restore trap occupancy levels """
        # Book keeping, of how many electrons have ended up where.
        # About to forget about those currently in traps, so add them to previous total.
        # About to recall the ones in save back to current account, so remove them
        # from the savings.
        self._n_electrons_trapped_previously += (
            self.n_electrons_trapped_currently - self._n_electrons_trapped_in_save
        )
        # Overwrite the current trap state
        if self._saved_data is None:
            self.empty_all_traps()
        else:
            self.data = deepcopy(self._saved_data)


class TrapManager(object):
    def __init__(self, traps, max_n_transfers):
        """
        The manager for potentially multiple trap species that are able to use 
        watermarks in the same way as each other.

        Parameters
        ----------
        traps : Trap or [Trap]
            A list of one or more trap species. Species listed together must be 
            able to share watermarks - i.e. they must be similarly distributed 
            throughout the pixel volume, and all their states must be stored 
            either by occupancy or by time since filling. 
            e.g. [bulk_trap_slow,bulk_trap_fast]
            
        max_n_transfers : int
            The number of pixels containing traps that charge will be expected 
            to move. This determines the maximum number of possible capture/
            release events that could create new watermark levels, and is used 
            to initialise the watermark array to be only as large as needed, for 
            efficiency.
                
        Attributes
        ----------
        watermarks : np.ndarray
            Array of watermark fractional volumes and fill fractions to describe 
            the trap states. Lists each (active) watermark fractional volume and 
            the corresponding fill fractions of each trap species. Inactive 
            elements are set to 0.

            [[volume, fill, fill, ...],
             [volume, fill, fill, ...],
             ...                       ]
             
        n_traps_per_pixel : np.ndarray
            The densities of all the trap species.
        
        capture_rates, emission_rates, total_rates : np.ndarray
            The rates of capture, emission, and their sum for all the traps.
        
        """
        if not isinstance(traps, list):
            traps = [traps]
        self.traps = deepcopy(traps)
        self._max_n_transfers = max_n_transfers

        # Set up the watermark array
        self.watermarks = np.zeros(
            (
                # This +1 is to ensure there is always at least one zero, even
                # if all transfers create a new watermark. The zero is used to
                # find the highest used watermark
                1 + self.max_n_transfers * self.n_watermarks_per_transfer,
                # This +1 is to make space for the volume column
                1 + self.n_trap_species,
            ),
            dtype=float,
        )

        # Trap rates
        self.capture_rates = np.array([trap.capture_rate for trap in traps])
        self.emission_rates = np.array([trap.emission_rate for trap in traps])
        self.total_rates = self.capture_rates + self.emission_rates

        # Are they surface traps?
        self.surface = np.array([trap.surface for trap in traps], dtype=bool)

        # Construct a function that describes the fraction of traps that are
        # exposed by a charge cloud containing n_electrons. This must be
        # fractional (rather than the absolute number of traps) because it can
        # be shared between trap species that have different trap densities.
        if self.traps[0].distribution_within_pixel() == None:

            def _fraction_of_traps_exposed_from_n_electrons(
                self, n_electrons, ccd_filling_function, surface=self.surface[0]
            ):
                return ccd_filling_function(n_electrons)

        else:

            def _fraction_of_traps_exposed_from_n_electrons(
                self, n_electrons, ccd_filling_function
            ):
                fraction_of_traps = 0
                #
                # RJM: RESERVED FOR SURFACE TRAPS OR SPECIES WITH NONUNIFORM DENSITY WITHIN A PIXEL
                #
                return fraction_of_traps

        self._fraction_of_traps_exposed_from_n_electrons = (
            _fraction_of_traps_exposed_from_n_electrons
        )

    def fraction_of_traps_exposed_from_n_electrons(
        self, n_electrons, ccd_filling_function
    ):
        """ Calculate the proportion of traps reached by a charge cloud. """
        return self._fraction_of_traps_exposed_from_n_electrons(
            self, n_electrons, ccd_filling_function
        )

    @property
    def filled_watermark_value(self):
        """ The value for a full watermark level, here 1 because we are 
            monitoring the fill fraction.
        """
        return 1

    @property
    def n_watermarks_per_transfer(self):
        """ Each transfer can create up to 2 new watermark levels (at the other 
            extreme, a transfer can remove all watermarks bar one) 
        """
        return 2

    @property
    def max_n_transfers(self):
        """ Number of pixels for which this manager is expected to monitor trap occupancy """
        return self._max_n_transfers

    @property
    def n_trap_species(self):
        """ Total number of trap species within this trap manager """
        return len(self.traps)

    @property
    def n_traps_per_pixel(self):
        """ Number of traps of each species, in each pixel """
        return np.array([trap.density for trap in self.traps], dtype=np.float64)

    @n_traps_per_pixel.setter
    def n_traps_per_pixel(self, values):
        if len(values) != len(self.traps):
            raise ValueError(
                f"{len(value)} trap densities supplied, for {len(self.traps)} species of traps"
            )
        for i, value in enumerate(values):
            self.traps[i].density = float(value)

    @property
    def delta_ellipticity(self):
        return sum([trap.delta_ellipticity for trap in self.traps])

    def fill_probabilities_from_dwell_time(self, dwell_time):
        """ The probabilities of being full after release and/or capture.
        
        See Lindegren (1998) section 3.2.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
            
        Returns
        -------
        fill_probabilities_from_empty : float
            The fraction of traps that were empty that become full.
            
        fill_probabilities_from_full : float
            The fraction of traps that were full that stay full.
            
        fill_probabilities_from_release : float
            The fraction of traps that were full that stay full after release.
        """
        # Common factor for capture and release probabilities
        exponential_factor = (
            1 - np.exp(-self.total_rates * dwell_time)
        ) / self.total_rates

        # New fill fraction for empty traps (Eqn. 20)
        # Ignore unnecessary warning from instant capture
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in multiply"
            )
            fill_probabilities_from_empty = self.capture_rates * exponential_factor
        # Fix for instant capture
        fill_probabilities_from_empty[np.isnan(fill_probabilities_from_empty)] = 1

        # New fill fraction for filled traps (Eqn. 21)
        fill_probabilities_from_full = 1 - self.emission_rates * exponential_factor

        # New fill fraction from only release
        fill_probabilities_from_release = np.exp(-self.emission_rates * dwell_time)

        return (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        )

    def n_trapped_electrons_from_watermarks(self, watermarks):
        """ Sum the total number of electrons currently held in traps.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        return cy_n_trapped_electrons_from_watermarks(
            watermarks, self.n_traps_per_pixel
        )

    def empty_all_traps(self):
        """ Reset the trap watermarks for the next run of release and capture. """
        self.watermarks.fill(0)

    def watermark_index_above_cloud_from_cloud_fractional_volume(
        self, cloud_fractional_volume, watermarks, max_watermark_index
    ):
        """ Return the index of the first watermark above the cloud.
            
        Parameters
        ----------
        cloud_fractional_volume : float
            The fractional volume of the electron cloud in the pixel.
            
        watermarks : np.ndarray
            The initial watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
            
        max_watermark_index : int
            The index of the highest existing watermark.
            
        Returns
        -------
        watermark_index_above_cloud : int
            The index of the first watermark above the cloud.
        """
        if cloud_fractional_volume == 0:
            return -1
        else:
            return cy_watermark_index_above_cloud_from_cloud_fractional_volume(
                cloud_fractional_volume, watermarks, max_watermark_index
            )

    def update_watermark_volumes_for_cloud_below_highest(
        self, watermarks, cloud_fractional_volume, watermark_index_above_cloud
    ):
        """ Update the trap watermarks for a cloud below the highest watermark.
            
        Parameters
        ----------
        watermarks : np.ndarray
            The initial watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
            
        cloud_fractional_volume : float
            The fractional volume of the electron cloud in the pixel.
            
        watermark_index_above_cloud : int
            The index of the first watermark above the cloud.
            
        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        cy_update_watermark_volumes_for_cloud_below_highest(
            watermarks, cloud_fractional_volume, watermark_index_above_cloud
        )
        return watermarks

    def updated_watermarks_from_capture_not_enough(
        self, watermarks, watermarks_initial, enough
    ):
        """ 
        Tweak trap watermarks for capturing electrons when not enough are 
        available to fill every trap below the cloud fractional volume (rare!).
        
        Parameters
        ----------
        watermarks : np.ndarray
            The current watermarks after attempted capture. See 
            initial_watermarks_from_n_pixels_and_total_traps().
            
        watermarks_initial : np.ndarray
            The initial watermarks before capture, but with updated fractional 
            volumes to match the current watermarks.
            
        enough : float
            The ratio of available electrons to traps up to this fractional 
            volume.

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """

        # Limit the increase to the `enough` fraction of the original
        watermarks[:, 1:] = (
            enough * watermarks[:, 1:] + (1 - enough) * watermarks_initial[:, 1:]
        )

        return watermarks

    def collapse_redundant_watermarks(self, watermarks, watermarks_copy=None):
        """ 
        Collapse any redundant watermarks that are completely full.
        
        Parameters
        ----------
        watermarks : np.ndarray
            The current watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().            
            
        watermarks_copy : np.ndarray
            A copy of the watermarks array that should be edited in the same way 
            as watermarks.

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().  
              
        watermarks_copy : np.ndarray
            The updated watermarks copy, if it was provided.
        """
        # Number of trap species
        num_traps = watermarks.shape[1] - 1

        # Find the first watermark that is not completely filled for all traps
        watermark_index_not_filled = min(
            [
                np.argmax(watermarks[:, 1 + i_trap] != self.filled_watermark_value)
                for i_trap in range(num_traps)
            ]
        )

        # Skip if none or only one are completely filled
        if watermark_index_not_filled <= 1:
            if watermarks_copy is not None:
                return watermarks, watermarks_copy
            else:
                return watermarks

        # Total fractional volume of filled watermarks
        fractional_volume_filled = np.sum(watermarks[:watermark_index_not_filled, 0])

        # Combined fill values
        if watermarks_copy is not None:
            # Multiple trap species
            if 1 < num_traps:
                axis = 1
            else:
                axis = None
            copy_fill_values = np.sum(
                watermarks_copy[:watermark_index_not_filled, 0]
                * watermarks_copy[:watermark_index_not_filled, 1:].T,
                axis=axis,
            ) / np.sum(watermarks_copy[:watermark_index_not_filled, 0])

        # Remove the no-longer-needed overwritten watermarks
        watermarks[:watermark_index_not_filled, :] = 0
        if watermarks_copy is not None:
            watermarks_copy[:watermark_index_not_filled, :] = 0

        # Move the no-longer-needed watermarks to the end of the list
        watermarks = np.roll(watermarks, 1 - watermark_index_not_filled, axis=0)
        if watermarks_copy is not None:
            watermarks_copy = np.roll(
                watermarks_copy, 1 - watermark_index_not_filled, axis=0
            )

        # Edit the new first watermark
        watermarks[0, 0] = fractional_volume_filled
        watermarks[0, 1:] = self.filled_watermark_value
        if watermarks_copy is not None:
            watermarks_copy[0, 0] = fractional_volume_filled
            watermarks_copy[0, 1:] = copy_fill_values

        if watermarks_copy is not None:
            return watermarks, watermarks_copy
        else:
            return watermarks

    def _n_electrons_released_and_captured__first_capture(
        self,
        n_free_electrons,
        cloud_fractional_volume,
        watermarks_initial,
        fill_probabilities_from_empty,
        express_multiplier,
    ):
        """ For n_electrons_released_and_captured(), for the first capture.
        
        Make the new watermark then can return immediately.
        """

        # Update the watermark volume, duplicated for the initial watermarks
        self.watermarks[0, 0] = cloud_fractional_volume
        watermarks_initial[0, 0] = self.watermarks[0, 0]

        # Update the fill fractions
        self.watermarks[0, 1:] = fill_probabilities_from_empty

        # Final number of electrons in traps
        n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Not enough available electrons to capture
        if n_trapped_electrons_final == 0:
            return 0

        enough = n_free_electrons / n_trapped_electrons_final
        if enough < 1:
            # For watermark fill fractions that increased, tweak them such that
            # the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                self.watermarks, watermarks_initial, enough
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
                watermarks=self.watermarks
            )

        return -n_trapped_electrons_final

    def _n_electrons_released_and_captured__cloud_below_watermarks(
        self,
        n_free_electrons,
        cloud_fractional_volume,
        watermarks_initial,
        watermark_index_above_cloud,
        max_watermark_index,
        fill_probabilities_from_release,
        n_trapped_electrons_initial,
        ccd_filling_function,
    ):
        """ For n_electrons_released_and_captured(), for capture from a cloud 
            below the existing watermarks.
        
        Create a new watermark at the cloud fractional volume then release 
        electrons from watermarks above the cloud.
        """
        # Create the new watermark at the cloud fractional volume
        if cloud_fractional_volume > 0 and not cy_value_in_cumsum(
            cloud_fractional_volume, self.watermarks[:, 0]
        ):

            # Update the watermark volumes, duplicated for the initial watermarks
            self.watermarks = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=self.watermarks,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )
            watermarks_initial = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=watermarks_initial,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )

            # Increment the index now that an extra watermark has been set
            max_watermark_index += 1

        # Release electrons from existing watermark levels above the cloud
        # Update the fill fractions
        self.watermarks[
            watermark_index_above_cloud + 1 : max_watermark_index + 1, 1:
        ] *= fill_probabilities_from_release

        # Current numbers of electrons temporarily in traps and now available
        n_trapped_electrons_tmp = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )
        n_free_electrons += n_trapped_electrons_initial - n_trapped_electrons_tmp

        # Re-calculate the fractional volume of the electron cloud
        cloud_fractional_volume = self.fraction_of_traps_exposed_from_n_electrons(
            n_electrons=n_free_electrons, ccd_filling_function=ccd_filling_function
        )
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # Update the watermark volumes, duplicated for the initial watermarks
        if cloud_fractional_volume > 0 and not cy_value_in_cumsum(
            cloud_fractional_volume, self.watermarks[:, 0]
        ):
            self.watermarks = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=self.watermarks,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )
            watermarks_initial = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=watermarks_initial,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )

            # Increment the index now that an extra watermark has been set
            max_watermark_index += 1

        return (
            n_free_electrons,
            watermarks_initial,
            cloud_fractional_volume,
            watermark_index_above_cloud,
            max_watermark_index,
        )

    def n_electrons_released_and_captured(
        self,
        n_free_electrons,
        ccd_filling_function,
        dwell_time=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.
        
        See Lindegren (1998) section 3.2.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
            
        ccd_filling_function : func
            A (self-contained) function describing the well-filling model, 
            returning the fractional volume of charge cloud in a pixel from 
            the number of electrons in the cloud.
            
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
            
        express_multiplier : int
            (Not currently used.)
        
            The number of times this transfer is to be replicated as part of the 
            express algorithm, passed here to make sure that too many electrons 
            are not removed if the multiplier is too high, to avoid ending up 
            with negative charge in the image.
            
        Returns
        -------
        net_n_electrons_released : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Initial watermarks and number of electrons in traps
        # watermarks_initial = deepcopy(self.watermarks)
        watermarks_initial = np.array(self.watermarks)
        n_trapped_electrons_initial = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Probabilities of being full after release and/or capture
        (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional volume the electron cloud reaches in the pixel well
        cloud_fractional_volume = self.fraction_of_traps_exposed_from_n_electrons(
            n_electrons=n_free_electrons, ccd_filling_function=ccd_filling_function
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # Proceed with capture depending on the volume of the cloud compared
        # with the existing watermark levels:

        # First capture (no existing watermarks)
        if max_watermark_index == -1 and n_free_electrons > 0:
            # Make the new watermark then can return immediately
            n_trapped_electrons_final = self._n_electrons_released_and_captured__first_capture(
                n_free_electrons,
                cloud_fractional_volume,
                watermarks_initial,
                fill_probabilities_from_empty,
                express_multiplier,
            )

            return n_trapped_electrons_final

        # Cloud fractional volume below existing watermarks (or 0)
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):
            # Create a new watermark at the cloud fractional volume then release
            # electrons from watermarks above the cloud
            (
                n_free_electrons,
                watermarks_initial,
                cloud_fractional_volume,
                watermark_index_above_cloud,
                max_watermark_index,
            ) = self._n_electrons_released_and_captured__cloud_below_watermarks(
                n_free_electrons,
                cloud_fractional_volume,
                watermarks_initial,
                watermark_index_above_cloud,
                max_watermark_index,
                fill_probabilities_from_release,
                n_trapped_electrons_initial,
                ccd_filling_function,
            )

        # Cloud fractional volume above existing watermarks
        else:
            # Initialise the new watermark, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = cloud_fractional_volume - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Continue with capture having prepared the new watermark at the cloud
        # and released from any watermarks above the cloud

        # Release and capture electrons all the way to watermarks below the cloud
        fill_fractions_old = self.watermarks[: watermark_index_above_cloud + 1, 1:]
        self.watermarks[: watermark_index_above_cloud + 1, 1:] = (
            fill_fractions_old * fill_probabilities_from_full
            + (1 - fill_fractions_old) * fill_probabilities_from_empty
        )

        # Collapse any redundant watermarks that are completely full
        self.watermarks, watermarks_initial = self.collapse_redundant_watermarks(
            watermarks=self.watermarks, watermarks_copy=watermarks_initial
        )

        # Final number of electrons in traps
        n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Prevent division by zero errors
        if n_trapped_electrons_final == n_trapped_electrons_initial:
            return 0

        # Not enough available electrons to capture
        enough = n_free_electrons / (
            n_trapped_electrons_final - n_trapped_electrons_initial
        )
        if 0 < enough < 1:
            # For watermark fill fractions that increased, tweak them such that
            # the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                watermarks=self.watermarks,
                watermarks_initial=watermarks_initial,
                enough=enough,
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
                watermarks=self.watermarks
            )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        return n_trapped_electrons_initial - n_trapped_electrons_final


class TrapManagerInstantCapture(TrapManager):
    """ For the old C++ style release-then-instant-capture algorithm. """

    @property
    def n_watermarks_per_transfer(self):
        return 1

    def update_watermark_values_for_release(self, watermark_values, dwell_time):
        """ 
        Update trap watermarks for releasing electrons.
        
        Parameters
        ----------
        watermark_values : np.ndarray
            The relevant subarray of the watermarks to update. i.e. not the 
            fractional volumes, and only the levels that are doing release. 

        Returns
        -------
        watermark_values : np.ndarray
            The updated watermark subarray.
        """
        # Probabilities of being full after release and/or capture
        (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Update the fill fractions
        return watermark_values * fill_probabilities_from_release

    def n_electrons_released(self, dwell_time=1):
        """ 
        Release electrons from traps and update the trap watermarks.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
            
        Returns
        -------
        n_electrons_released : float
            The number of released electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Initial watermarks and number of electrons in traps
        n_trapped_electrons_initial = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # Release electrons from existing watermark levels
        self.watermarks[
            : max_watermark_index + 1, 1:
        ] = self.update_watermark_values_for_release(
            self.watermarks[: max_watermark_index + 1, 1:], dwell_time,
        )

        # Resulting numbers of electrons in traps
        n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Total number of released electrons and updated free electrons
        n_electrons_released = n_trapped_electrons_initial - n_trapped_electrons_final

        return n_trapped_electrons_initial - n_trapped_electrons_final

    def _n_electrons_captured__first_capture(
        self,
        n_free_electrons,
        cloud_fractional_volume,
        watermarks_initial,
        express_multiplier,
    ):
        """ For n_electrons_captured(), for the first capture.
        
        Make the new watermark then can return immediately.
        """
        # Update the watermark volume, duplicated for the initial watermarks
        self.watermarks[0, 0] = cloud_fractional_volume
        watermarks_initial[0, 0] = self.watermarks[0, 0]

        # Update the fill fractions
        self.watermarks[0, 1:] = self.filled_watermark_value

        # Final number of electrons in traps
        n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Not enough available electrons to capture
        if n_trapped_electrons_final == 0:
            return 0.0
        # Also ensure the final number of electrons in the pixel is positive
        enough = n_free_electrons / n_trapped_electrons_final
        if enough < 1:
            # For watermark fill fractions that increased, tweak them such
            # that the resulting increase instead matches the available
            # electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                self.watermarks, watermarks_initial, enough
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
                watermarks=self.watermarks
            )

        return n_trapped_electrons_final

    def _n_electrons_captured__cloud_below_watermarks(
        self,
        cloud_fractional_volume,
        watermarks_initial,
        watermark_index_above_cloud,
        max_watermark_index,
    ):
        """ For n_electrons_captured(), for capture from a cloud below the 
            existing watermarks.
        
        Create a new watermark at the cloud fractional volume.
        """
        # Create the new watermark at the cloud fractional volume
        if cloud_fractional_volume > 0 and cloud_fractional_volume not in np.cumsum(
            self.watermarks[:, 0]
        ):
            # Update the watermark volumes, duplicated for the initial watermarks
            self.watermarks = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=self.watermarks,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )
            watermarks_initial = self.update_watermark_volumes_for_cloud_below_highest(
                watermarks=watermarks_initial,
                cloud_fractional_volume=cloud_fractional_volume,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )

            # Increment the index now that an extra watermark has been set
            max_watermark_index += 1

        return watermarks_initial, max_watermark_index

    def n_electrons_captured(
        self,
        n_free_electrons,
        ccd_filling_function,
        dwell_time=1,
        express_multiplier=1,
    ):
        """
        Capture electrons in traps and update the trap watermarks.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
            
        ccd_filling_function : func
            A (self-contained) function describing the well-filling model, 
            returning the fractional volume of charge cloud in a pixel from 
            the number of electrons in the cloud.
            
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
            
        express_multiplier : int
            (Not currently used.)
        
            The number of times this transfer is to be replicated as part of the 
            express algorithm, passed here to make sure that too many electrons 
            are not removed if the multiplier is too high, to avoid ending up 
            with negative charge in the image.

        Returns
        -------
        n_electrons_captured : float
            The number of captured electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Save the initial watermarks and number of electrons in traps, in case
        # there aren't enough free electrons to fill all the identified traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional volume the electron cloud reaches in the pixel well
        cloud_fractional_volume = self.fraction_of_traps_exposed_from_n_electrons(
            n_electrons=n_free_electrons, ccd_filling_function=ccd_filling_function
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # Proceed with capture depending on the volume of the cloud compared
        # with the existing watermark levels:

        # First capture (no existing watermarks)
        if max_watermark_index == -1 and n_free_electrons > 0:
            # Make the new watermark then can return immediately
            n_trapped_electrons_final = self._n_electrons_captured__first_capture(
                n_free_electrons,
                cloud_fractional_volume,
                watermarks_initial,
                express_multiplier,
            )

            return n_trapped_electrons_final

        # Cloud fractional volume below existing watermarks (or 0)
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):
            # Create a new watermark at the cloud fractional volume
            (
                watermarks_initial,
                max_watermark_index,
            ) = self._n_electrons_captured__cloud_below_watermarks(
                cloud_fractional_volume,
                watermarks_initial,
                watermark_index_above_cloud,
                max_watermark_index,
            )

        # Cloud fractional volume above existing watermarks
        else:
            # Initialise the new watermark, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = cloud_fractional_volume - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Continue with capture having prepared the new watermark at the cloud
        # and released from any watermarks above the cloud

        # Capture electrons all the way to watermarks below the cloud
        self.watermarks[
            : watermark_index_above_cloud + 1, 1:
        ] = self.filled_watermark_value

        # Collapse any redundant watermarks that are completely full
        self.watermarks, watermarks_initial = self.collapse_redundant_watermarks(
            watermarks=self.watermarks, watermarks_copy=watermarks_initial
        )

        # Final number of electrons in traps
        n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
            watermarks=self.watermarks
        )

        # Prevent division by zero errors
        if n_trapped_electrons_final == n_trapped_electrons_initial:
            return 0.0

        # Not enough available electrons to capture
        enough = n_free_electrons / (
            n_trapped_electrons_final - n_trapped_electrons_initial
        )
        if 0 < enough < 1:
            # For watermark fill fractions that increased, tweak them such that
            # the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                watermarks=self.watermarks,
                watermarks_initial=watermarks_initial,
                enough=enough,
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.n_trapped_electrons_from_watermarks(
                watermarks=self.watermarks
            )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        return n_trapped_electrons_final - n_trapped_electrons_initial

    def n_electrons_released_and_captured(
        self,
        n_free_electrons,
        ccd_filling_function,
        dwell_time=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
            
        ccd_filling_function : func
            A (self-contained) function describing the well-filling model, 
            returning the fractional volume of charge cloud in a pixel from 
            the number of electrons in the cloud.
            
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
            
        express_multiplier : int
            (Not currently used.)
        
            The number of times this transfer is to be replicated as part of the 
            express algorithm, passed here to make sure that too many electrons 
            are not removed if the multiplier is too high, to avoid ending up 
            with negative charge in the image.
            
        Returns
        -------
        net_n_electrons_released_and_captured : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Release
        n_electrons_released = self.n_electrons_released(dwell_time=dwell_time)
        n_free_electrons += n_electrons_released

        # Capture
        n_electrons_captured = self.n_electrons_captured(
            n_free_electrons,
            ccd_filling_function=ccd_filling_function,
            express_multiplier=express_multiplier,
        )

        return n_electrons_released - n_electrons_captured


class TrapManagerTrackTime(TrapManagerInstantCapture):
    """ Track the time elapsed since capture instead of the fill fraction. 
    
    Should give the same result for normal traps with TrapManagerInstantCapture, 
    and required for TrapLifetimeContinuumAbstract traps.
    
    Note the different watermark contents:        
    watermarks : np.ndarray
        Array of watermark fractional volumes and times to describe the trap 
        states. Lists each (active) watermark fractional volume and and the 
        corresponding total time elapsed since the traps were filled for 
        each trap species. Inactive elements are set to 0.

        [[volume, time_elapsed, time_elapsed, ...],
         [volume, time_elapsed, time_elapsed, ...],
         ...                                       ]
    """

    @property
    def filled_watermark_value(self):
        """ The value for a filled watermark level, here 0 is an elapsed time """
        return 0

    def watermarks_converted_to_fill_fractions_from_elapsed_times(self, watermarks):
        """ Convert the watermark values to fill fractions.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        watermarks = deepcopy(watermarks)

        # Convert to fill fractions
        for level in range(np.argmax(watermarks[:, 0] == 0)):
            for i_trap, trap in enumerate(self.traps):
                watermarks[level, 1 + i_trap] = trap.fill_fraction_from_time_elapsed(
                    time_elapsed=watermarks[level, 1 + i_trap]
                )

        return watermarks

    def watermarks_converted_to_elapsed_times_from_fill_fractions(self, watermarks):
        """ Convert the watermark values to elapsed times.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        watermarks = deepcopy(watermarks)

        # Convert to elapsed times
        for level in range(np.argmax(watermarks[:, 0] == 0)):
            for i_trap, trap in enumerate(self.traps):
                watermarks[level, 1 + i_trap] = trap.time_elapsed_from_fill_fraction(
                    fill_fraction=watermarks[level, 1 + i_trap]
                )

        return watermarks

    def n_trapped_electrons_from_watermarks(self, watermarks):
        """ Sum the total number of electrons currently held in traps.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Convert to fill fractions
        watermarks = self.watermarks_converted_to_fill_fractions_from_elapsed_times(
            watermarks=watermarks
        )
        return cy_n_trapped_electrons_from_watermarks(
            watermarks, self.n_traps_per_pixel
        )

    def updated_watermarks_from_capture_not_enough(
        self, watermarks, watermarks_initial, enough
    ):
        """ 
        Tweak trap watermarks for capturing electrons when not enough are 
        available to fill every trap below the cloud fractional volume (rare!).
        
        Parameters
        ----------
        watermarks : np.ndarray
            The current watermarks after attempted capture. See 
            initial_watermarks_from_n_pixels_and_total_traps().
            
        watermarks_initial : np.ndarray
            The initial watermarks before capture, but with updated fractional 
            volumes to match the current watermarks.
            
        enough : float
            The ratio of available electrons to traps up to this fractional 
            volume.

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See 
            initial_watermarks_from_n_pixels_and_total_traps().
        """
        # Convert to fill fractions
        for level in range(np.argmax(watermarks[:, 0] == 0)):
            for i_trap, trap in enumerate(self.traps):
                watermarks[level, i_trap] = trap.fill_fraction_from_time_elapsed(
                    time_elapsed=watermarks[level, i_trap]
                )
                watermarks_initial[
                    level, i_trap
                ] = trap.fill_fraction_from_time_elapsed(
                    time_elapsed=watermarks_initial[level, i_trap]
                )

        # Update the fill fractions in the standard way
        watermarks = super(
            TrapManagerTrackTime, self
        ).updated_watermarks_from_capture_not_enough(
            self,
            watermarks=watermarks,
            watermarks_initial=watermarks_initial,
            enough=enough,
        )

        # Convert back to elapsed times
        for level in range(np.argmax(watermarks[:, 0] == 0)):
            for i_trap, trap in enumerate(self.traps):
                watermarks[level, i_trap] = trap.time_elapsed_from_fill_fraction(
                    fill_fraction=watermarks[level, i_trap]
                )

        return watermarks

    def update_watermark_values_for_release(self, watermark_values, dwell_time):
        """ 
        Update trap watermarks for releasing electrons.
        
        Parameters
        ----------
        watermark_values : np.ndarray
            The relevant subarray of the watermarks to update. i.e. not the 
            fractional volumes, and only the elapsed times of the levels that 
            are doing release. 

        Returns
        -------
        watermark_values : np.ndarray
            The updated watermark subarray.
        """
        # Update the elapsed times
        return watermark_values + dwell_time
