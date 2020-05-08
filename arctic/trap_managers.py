import numpy as np
from scipy import integrate, optimize
from copy import deepcopy


class TrapManager(object):
    def __init__(self, traps, rows):
        """The manager for potentially multiple trap species that must use 
        watermarks in the same way as each other.

        Parameters
        ----------
        traps : [Trap]
            A list of one or more trap objects.
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
    
        Attributes
        ----------
        watermarks : np.ndarray
            The watermarks. See initial_watermarks_from_rows_and_total_traps().
        densities : np.ndarray
            The densities of all the trap species.
        capture_rates, emission_rates, total_rates : np.ndarray
            The rates of capture, emission, and their sum for all the traps.
        """
        self.traps = traps
        self.rows = rows

        # Set up the watermarks
        self.watermarks = self.initial_watermarks_from_rows_and_total_traps(
            rows=self.rows, total_traps=len(self.traps)
        )

        # Trap densities
        self.densities = np.array([trap.density for trap in self.traps])

        # Trap rates
        self.capture_rates = np.array([trap.capture_rate for trap in self.traps])
        self.emission_rates = np.array([trap.emission_rate for trap in self.traps])
        self.total_rates = self.capture_rates + self.emission_rates

    @property
    def delta_ellipticity(self):
        return sum([trap.delta_ellipticity for trap in self.traps])

    def initial_watermarks_from_rows_and_total_traps(self, rows, total_traps):
        """ Initialise the watermark array of trap states.

        Parameters
        ----------
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
        total_traps : int
            The number of traps being modelled.

        Returns
        -------
        watermarks : np.ndarray
            Array of watermark heights and fill fractions to describe the trap states. Lists each (active) watermark
            height_e fraction and the corresponding fill fractions of each traps. Inactive elements are set to 0.

            [[height_e, fill, fill, ...],
             [height_e, fill, fill, ...],
             ...                       ]
        """
        return np.zeros((rows * 2, 1 + total_traps), dtype=float)

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

    def fraction_of_exposed_traps(self, n_free_electrons, ccd):
        """ 
        Calculate the total number of charge traps exposed to a charge cloud
        containing n_electrons. This assumes that charge traps are uniformly
        distributed through the volume, but that assumption can be relaxed
        by adjusting this function to reflect the net number of traps seen
        as a function of charge cloud size. An example of that is provided,
        for surface traps that are responsible for blooming (which is
        asymmetric and happens during readout, unlike bleeding). 
        
        This function embodies the core assumption of a volume-driven CTI
        model like arCTIc: that traps are either exposed (and have a
        constant capture timescale, which may be zero for instant capture),
        or unexposed and therefore unavailable. This behaviour differs from
        a density-driven CTI model, in which traps may capture an electron
        anywhere in a pixel, but at varying capture probability. There is 
        considerable evidence that CCDs in the Hubble Space Telescope are 
        primarily density-driven; a software algorithm to mimic such 
        behaviour also runs much faster.
        
        
        """
        if self.traps[0].surface:
            n_exposed_traps = max(n_free_electrons - ccd.blooming_level, 0)
            # RJM: what density to use if more than one trap species?
            fraction_of_exposed_traps = min(n_exposed_traps / self.densities, 1)
            if 0 < n_exposed_traps < 1:
                print(
                    "hello",
                    n_free_electrons,
                    ccd.blooming_level,
                    fraction_of_exposed_traps,
                )
            # fraction_of_exposed_traps = n_exposed_traps / self.densities

        else:
            fraction_of_exposed_traps = ccd.cloud_fractional_volume_from_electrons(
                electrons=n_free_electrons
            )

        return fraction_of_exposed_traps

    def number_of_trapped_electrons_from_watermarks(
        self, watermarks, fractional_width=1
    ):
        """ Sum the total number of electrons currently held in traps.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See initial_watermarks_from_rows_and_total_traps().
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
        """
        return (
            np.sum((watermarks[:, 0] * watermarks[:, 1:].T).T * self.densities)
            * fractional_width
        )

    def empty_all_traps(self):
        """Reset the trap watermarks for the next run of release and capture.
        """
        self.watermarks.fill(0)

    def watermark_index_above_cloud_from_cloud_fractional_volume(
        self, cloud_fractional_volume, watermarks, max_watermark_index
    ):
        """ Return the index of the first watermark above the cloud.
            
        Parameters
        ----------
        cloud_fractional_volume : float
            The fractional height of the electron cloud in the pixel.
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        max_watermark_index : int
            The index of the highest existing watermark.
            
        Returns
        -------
        watermark_index_above_cloud : int
            The index of the first watermark above the cloud.
        """
        if np.sum(watermarks[:, 0]) < cloud_fractional_volume:
            return max_watermark_index + 1

        elif cloud_fractional_volume == 0:
            return -1

        else:
            return np.argmax(cloud_fractional_volume < np.cumsum(watermarks[:, 0]))

    def update_watermark_heights_for_cloud_below_highest(
        self, watermarks, cloud_fractional_volume, watermark_index_above_cloud
    ):
        """ Update the trap watermarks for a cloud below the highest watermark.
            
        Parameters
        ----------
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        cloud_fractional_volume : float
            The fractional height of the electron cloud in the pixel.
        watermark_index_above_cloud : int
            The index of the first watermark above the cloud.
            
        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # The height and cumulative height of the watermark around the cloud height
        watermark_height = watermarks[watermark_index_above_cloud, 0]
        cumulative_watermark_height = np.sum(
            watermarks[: watermark_index_above_cloud + 1, 0]
        )

        # Move one new empty watermark to the start of the list
        watermarks = np.roll(watermarks, 1, axis=0)

        # Re-set the relevant watermarks near the start of the list
        if watermark_index_above_cloud == 0:
            watermarks[0] = watermarks[1]
        else:
            watermarks[: watermark_index_above_cloud + 1] = watermarks[
                1 : watermark_index_above_cloud + 2
            ]

        # Update the new split watermarks' heights
        old_height = watermark_height
        watermarks[watermark_index_above_cloud, 0] = cloud_fractional_volume - (
            cumulative_watermark_height - watermark_height
        )
        watermarks[watermark_index_above_cloud + 1, 0] = (
            old_height - watermarks[watermark_index_above_cloud, 0]
        )

        return watermarks

    def updated_watermarks_from_capture_not_enough(
        self, watermarks, watermarks_initial, enough
    ):
        """ 
        Tweak trap watermarks for capturing electrons when not enough are 
        available to fill every trap below the cloud height (rare!).
        
        Parameters
        ----------
        watermarks : np.ndarray
            The current watermarks after attempted capture. See 
            initial_watermarks_from_rows_and_total_traps().
        watermarks_initial : np.ndarray
            The initial watermarks before capture, but with updated heights to 
            match the current watermarks.
        enough : float
            The ratio of available electrons to traps up to this height.

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Matching watermark heights
        assert (watermarks_initial[:, 0] == watermarks[:, 0]).all()

        # Select watermark fill fractions that increased
        where_increased = np.where(watermarks_initial[:, 1:] < watermarks[:, 1:])

        # Limit the increase to the `enough` fraction of the original
        watermarks[:, 1:] = (
            enough * watermarks[:, 1:] + (1 - enough) * watermarks_initial[:, 1:]
        )

        return watermarks

    def collapse_redundant_watermarks(self, watermarks):
        """ 
        Collapse any redundant watermarks that are completely full.
        
        Parameters
        ----------
        watermarks : np.ndarray
            The current watermarks. See initial_watermarks_from_rows_and_total_traps().

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Find the first watermark that is not completely filled
        watermark_index_not_filled = np.argmax(watermarks[:, 1] < 1)

        # Skip if none are completely filled
        if watermark_index_not_filled <= 1:
            return watermarks

        # Total height of filled watermarks
        height_filled = np.sum(watermarks[:watermark_index_not_filled, 0])

        # Remove the no-longer-needed overwritten watermarks
        watermarks[:watermark_index_not_filled, :] = 0

        # Move the no-longer-needed watermarks to the end of the list
        watermarks = np.roll(watermarks, 1 - watermark_index_not_filled, axis=0)

        # Edit the new first watermark
        watermarks[0, 0] = height_filled
        watermarks[0, 1:] = 1

        return watermarks

    def n_electrons_released_and_captured(
        self,
        n_free_electrons,
        ccd,
        dwell_time=1,
        fractional_width=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.
        
        See Lindegren (1998) section 3.2.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
        ccd : CCD
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCD.extract_phase().
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        net_n_electrons_released : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # The number of traps for each species
        densities = self.densities * fractional_width

        # Probabilities of being full after release and/or capture
        (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional height the electron cloud reaches in the pixel well
        cloud_fractional_volume = ccd.cloud_fractional_volume_from_electrons(
            electrons=n_free_electrons
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # First capture: make the new watermark then can return immediately
        if max_watermark_index == -1 and n_free_electrons > 0:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[0, 0] = cloud_fractional_volume
            watermarks_initial[0, 0] = self.watermarks[0, 0]

            # Update the fill fractions
            self.watermarks[0, 1:] = fill_probabilities_from_empty

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

            # Not enough available electrons to capture
            if n_trapped_electrons_final == 0:
                return 0
            enough = n_free_electrons / n_trapped_electrons_final
            if enough < 1:
                # For watermark fill fractions that increased, tweak them such that
                #   the resulting increase instead matches the available electrons
                self.watermarks = self.updated_watermarks_from_capture_not_enough(
                    self.watermarks, watermarks_initial, enough
                )

                # Final number of electrons in traps
                n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                    watermarks=self.watermarks, fractional_width=fractional_width
                )

            return -n_trapped_electrons_final

        # Cloud height below existing watermarks: create a new watermark at the
        #   cloud height then release electrons from watermarks above the cloud
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):

            # Create the new watermark at the cloud height
            if cloud_fractional_volume > 0:

                # Update the watermark heights, duplicated for the initial watermarks
                self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=self.watermarks,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )
                watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
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
            ] = (
                self.watermarks[
                    watermark_index_above_cloud + 1 : max_watermark_index + 1, 1:
                ]
                * fill_probabilities_from_release
            )

            # Current numbers of electrons temporarily in traps and now available
            n_trapped_electrons_tmp = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )
            n_free_electrons += n_trapped_electrons_initial - n_trapped_electrons_tmp

            # Re-calculate the height of the electron cloud
            cloud_fractional_volume = ccd.cloud_fractional_volume_from_electrons(
                electrons=n_free_electrons
            )
            watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
                cloud_fractional_volume=cloud_fractional_volume,
                watermarks=self.watermarks,
                max_watermark_index=max_watermark_index,
            )

            # Update the watermark heights, duplicated for the initial watermarks
            if cloud_fractional_volume > 0:
                self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=self.watermarks,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )
                watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=watermarks_initial,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )

                # Increment the index now that an extra watermark has been set
                max_watermark_index += 1

        # Cloud height above existing watermarks: initialise the new watermark
        else:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = cloud_fractional_volume - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Release and capture electrons all the way to watermarks below the cloud
        fill_fractions_old = self.watermarks[: watermark_index_above_cloud + 1, 1:]
        self.watermarks[: watermark_index_above_cloud + 1, 1:] = (
            fill_fractions_old * fill_probabilities_from_full
            + (1 - fill_fractions_old) * fill_probabilities_from_empty
        )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        # Final number of electrons in traps
        n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
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
            #   the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                watermarks=self.watermarks,
                watermarks_initial=watermarks_initial,
                enough=enough,
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        return n_trapped_electrons_initial - n_trapped_electrons_final


class TrapManagerInstantCapture(TrapManager):
    """ For the old C++ style release-then-instant-capture algorithm. """

    def initial_watermarks_from_rows_and_total_traps(self, rows, total_traps):
        """ Initialise the watermark array of trap states.

        Parameters
        ----------
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
        total_traps : int
            The number of traps being modelled.

        Returns
        -------
        watermarks : np.ndarray
            Array of watermark heights and fill fractions to describe the trap states. Lists each (active) watermark
            height_e fraction and the corresponding fill fractions of each traps. Inactive elements are set to 0.

            [[height_e, fill, fill, ...],
             [height_e, fill, fill, ...],
             ...                       ]
        """
        return np.zeros((rows, 1 + total_traps), dtype=float)

    def n_electrons_released(
        self, dwell_time=1, fractional_width=1, express_multiplier=1
    ):
        """ DEPRECATED but is used for some comparison tests. Replaced by 
            n_electrons_released_and_captured()
            
        Release electrons from traps and update the trap watermarks.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        n_electrons_released : float
            The number of released electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # The number of traps for each species
        densities = self.densities * fractional_width

        # Probabilities of being full after release and/or capture
        (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # Release electrons from existing watermark levels
        # Update the fill fractions
        self.watermarks[: max_watermark_index + 1, 1:] = (
            self.watermarks[: max_watermark_index + 1, 1:]
            * fill_probabilities_from_release
        )

        # Resulting numbers of electrons in traps
        n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        return n_trapped_electrons_initial - n_trapped_electrons_final

    def n_electrons_captured(
        self, n_free_electrons, ccd, fractional_width=1, express_multiplier=1
    ):
        """ DEPRECATED but is used for some comparison tests. Replaced by 
            n_electrons_released_and_captured()
            
        Capture electrons in traps and update the trap watermarks.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
        ccd : CCD
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCD.extract_phase().
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.

        Returns
        -------
        n_electrons_captured : float
            The number of captured electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # The number of traps for each species
        densities = self.densities * fractional_width

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional height the electron cloud reaches in the pixel well
        cloud_fractional_volume = ccd.cloud_fractional_volume_from_electrons(
            electrons=n_free_electrons
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # First capture: make the new watermark then can return immediately
        if max_watermark_index == -1 and n_free_electrons > 0:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[0, 0] = cloud_fractional_volume
            watermarks_initial[0, 0] = self.watermarks[0, 0]

            # Update the fill fractions
            self.watermarks[0, 1:] = 1

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

            # Not enough available electrons to capture
            if n_trapped_electrons_final == 0:
                return 0
            enough = n_free_electrons / n_trapped_electrons_final
            if enough < 1:
                # For watermark fill fractions that increased, tweak them such that
                #   the resulting increase instead matches the available electrons
                self.watermarks = self.updated_watermarks_from_capture_not_enough(
                    self.watermarks, watermarks_initial, enough
                )

                # Final number of electrons in traps
                n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                    watermarks=self.watermarks, fractional_width=fractional_width
                )

            return n_trapped_electrons_final

        # Cloud height below existing watermarks: create a new watermark at the
        #   cloud height
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):

            # Create the new watermark at the cloud height
            if cloud_fractional_volume > 0:

                # Update the watermark heights, duplicated for the initial watermarks
                self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=self.watermarks,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )
                watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=watermarks_initial,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )

                # Increment the index now that an extra watermark has been set
                max_watermark_index += 1

        # Cloud height above existing watermarks: initialise the new watermark
        else:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = cloud_fractional_volume - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Capture electrons all the way to watermarks below the cloud
        fill_fractions_old = self.watermarks[: watermark_index_above_cloud + 1, 1:]
        self.watermarks[: watermark_index_above_cloud + 1, 1:] = 1

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        # Final number of electrons in traps
        n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
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
            #   the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                watermarks=self.watermarks,
                watermarks_initial=watermarks_initial,
                enough=enough,
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        return n_trapped_electrons_final - n_trapped_electrons_initial

    def n_electrons_released_and_captured(
        self,
        n_free_electrons,
        ccd,
        dwell_time=1,
        fractional_width=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
        ccd : CCD
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCD.extract_phase().
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        net_n_electrons_released : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Release
        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # The number of traps for each species
        densities = self.densities * fractional_width

        # Probabilities of being full after release and/or capture
        (
            fill_probabilities_from_empty,
            fill_probabilities_from_full,
            fill_probabilities_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # Release electrons from existing watermark levels
        # Update the fill fractions
        self.watermarks[: max_watermark_index + 1, 1:] = (
            self.watermarks[: max_watermark_index + 1, 1:]
            * fill_probabilities_from_release
        )

        # Resulting numbers of electrons in traps
        n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # Total number of released electrons and updated free electrons
        n_electrons_released = n_trapped_electrons_initial - n_trapped_electrons_final
        n_free_electrons += n_electrons_released

        # Capture
        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        n_trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # The number of traps for each species
        densities = self.densities * fractional_width

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional height the electron cloud reaches in the pixel well
        cloud_fractional_volume = ccd.cloud_fractional_volume_from_electrons(
            electrons=n_free_electrons
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_cloud_fractional_volume(
            cloud_fractional_volume=cloud_fractional_volume,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # First capture: make the new watermark then can return immediately
        if max_watermark_index == -1 and n_free_electrons > 0:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[0, 0] = cloud_fractional_volume
            watermarks_initial[0, 0] = self.watermarks[0, 0]

            # Update the fill fractions
            self.watermarks[0, 1:] = 1

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

            # Not enough available electrons to capture
            if n_trapped_electrons_final == 0:
                return n_electrons_released
            enough = n_free_electrons / n_trapped_electrons_final
            if enough < 1:
                # For watermark fill fractions that increased, tweak them such that
                #   the resulting increase instead matches the available electrons
                self.watermarks = self.updated_watermarks_from_capture_not_enough(
                    self.watermarks, watermarks_initial, enough
                )

                # Final number of electrons in traps
                n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                    watermarks=self.watermarks, fractional_width=fractional_width
                )

            return n_electrons_released - n_trapped_electrons_final

        # Cloud height below existing watermarks: create a new watermark at the
        #   cloud height
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):

            # Create the new watermark at the cloud height
            if cloud_fractional_volume > 0:

                # Update the watermark heights, duplicated for the initial watermarks
                self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=self.watermarks,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )
                watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=watermarks_initial,
                    cloud_fractional_volume=cloud_fractional_volume,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )

                # Increment the index now that an extra watermark has been set
                max_watermark_index += 1

        # Cloud height above existing watermarks: initialise the new watermark
        else:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = cloud_fractional_volume - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Capture electrons all the way to watermarks below the cloud
        fill_fractions_old = self.watermarks[: watermark_index_above_cloud + 1, 1:]
        self.watermarks[: watermark_index_above_cloud + 1, 1:] = 1

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        # Final number of electrons in traps
        n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, fractional_width=fractional_width
        )

        # Prevent division by zero errors
        if n_trapped_electrons_final == n_trapped_electrons_initial:
            return n_electrons_released

        # Not enough available electrons to capture
        enough = n_free_electrons / (
            n_trapped_electrons_final - n_trapped_electrons_initial
        )
        if 0 < enough < 1:
            # For watermark fill fractions that increased, tweak them such that
            #   the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                watermarks=self.watermarks,
                watermarks_initial=watermarks_initial,
                enough=enough,
            )

            # Final number of electrons in traps
            n_trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, fractional_width=fractional_width
            )

        # Collapse any redundant watermarks that are completely full
        self.watermarks = self.collapse_redundant_watermarks(watermarks=self.watermarks)

        # Total number of captured electrons
        n_electrons_captured = n_trapped_electrons_final - n_trapped_electrons_initial

        return n_electrons_released - n_electrons_captured


class TrapManagerTrackTime(TrapManagerInstantCapture):
    """ Track the time elapsed since capture instead of the fill fraction. 
        Should give the same result for normal traps.
    """

    def initial_watermarks_from_rows_and_total_traps(self, rows, total_traps):
        """ Initialise the watermark array of trap states.

        Parameters
        ----------
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
        total_traps : int
            The number of traps being modelled.

        Returns
        -------
        watermarks : np.ndarray
            Array of watermark heights and times to describe the trap states. 
            Lists each (active) watermark height_e fraction and the 
            corresponding total time elapsed since the traps were filled for 
            each trap species. Inactive elements are set to 0.

            [[height_e, time_elapsed, time_elapsed, ...],
             [height_e, time_elapsed, time_elapsed, ...],
             ...                        ]
        """
        return np.zeros((rows, 1 + total_traps), dtype=float)

    def n_electrons_released(
        self, dwell_time=1, fractional_width=1, express_multiplier=1
    ):
        """ Release electrons from traps and update the trap watermarks.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        n_electrons_released : float
            The number of released electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initialise the number of released electrons
        n_electrons_released = 0

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # For each watermark
        for watermark_index in range(max_watermark_index + 1):
            # Initialise the number of released electrons from this watermark level
            n_electrons_released_watermark = 0

            # For each trap species
            for trap_index, trap in enumerate(self.traps):
                # Number of released electrons (not yet including the trap density)
                n_electrons_released_from_trap = trap.electrons_released_from_time_elapsed_and_dwell_time(
                    time_elapsed=self.watermarks[watermark_index, 1 + trap_index],
                    dwell_time=dwell_time,
                )

                # Update the watermark times
                self.watermarks[watermark_index, 1 + trap_index] += dwell_time

                # Update the actual number of released electrons
                n_electrons_released_watermark += (
                    n_electrons_released_from_trap * trap.density * fractional_width
                )

            # Multiply the summed fill fractions by the height
            n_electrons_released += (
                n_electrons_released_watermark * self.watermarks[watermark_index, 0]
            )

        return n_electrons_released

    def n_electrons_captured_by_traps(
        self, cloud_fractional_volume, watermarks, traps, fractional_width=1
    ):
        """
        Find the total number of electrons that the traps can capture.

        Parameters
        -----------
        cloud_fractional_volume : float
            The fractional height of the electron cloud in the pixel.
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        traps : [TrapSpecies]
            An array of one or more objects describing a trap of trap.
        fractional_width : float
            The width of this pixel or phase, as a fraction of the whole pixel.

        Returns
        -------
        n_electrons_captured : float
            The number of captured electrons.
        """
        # Initialise the number of captured electrons
        n_electrons_captured = 0

        # The number of traps of each species
        densities = np.array([trap.density for trap in traps]) * fractional_width

        # Find the highest active watermark
        max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1

        # Initialise cumulative watermark height
        cumulative_watermark_height = 0

        # Capture electrons above each existing watermark level below the highest
        for watermark_index in range(max_watermark_index + 1):
            # Update cumulative watermark height
            watermark_height = watermarks[watermark_index, 0]
            cumulative_watermark_height += watermark_height

            # Fill fractions from elapsed times
            fill_fractions = np.array(
                [
                    trap.fill_fraction_from_time_elapsed(time_elapsed)
                    for trap, time_elapsed in zip(
                        traps, watermarks[watermark_index, 1:]
                    )
                ]
            )

            # Capture electrons all the way to this watermark (for all traps)
            if cumulative_watermark_height < cloud_fractional_volume:
                n_electrons_captured += watermark_height * np.sum(
                    (1 - fill_fractions) * densities
                )

            # Capture electrons part-way between the previous and this watermark
            else:
                n_electrons_captured += (
                    cloud_fractional_volume
                    - (cumulative_watermark_height - watermark_height)
                ) * np.sum((1 - fill_fractions) * densities)

                # No point in checking even higher watermarks
                return n_electrons_captured

        # Capture any electrons above the highest existing watermark
        if watermarks[max_watermark_index, 0] < cloud_fractional_volume:
            n_electrons_captured += (
                cloud_fractional_volume - cumulative_watermark_height
            ) * np.sum(densities)

        return n_electrons_captured

    def updated_watermarks_from_capture(self, cloud_fractional_volume, watermarks):
        """ Update the trap watermarks for capturing electrons.

        Parameters
        ----------
        cloud_fractional_volume : float
            The fractional height of the electron cloud in the pixel.

        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
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
        if cumulative_watermark_height[-1] < cloud_fractional_volume:
            # Overwrite the first watermark
            watermarks[0, 0] = cloud_fractional_volume
            watermarks[0, 1:] = 0

            # Remove all higher watermarks
            watermarks[1:, :] = 0

            return watermarks

        # Find the first watermark above the cloud, which won't be fully overwritten
        watermark_index_above_cloud = np.argmax(
            cloud_fractional_volume < cumulative_watermark_height
        )

        # If some will be overwritten
        if 0 < watermark_index_above_cloud:
            # Edit the partially overwritten watermark
            watermarks[watermark_index_above_cloud, 0] -= (
                cloud_fractional_volume
                - cumulative_watermark_height[watermark_index_above_cloud - 1]
            )

            # Remove the no-longer-needed overwritten watermarks
            watermarks[: watermark_index_above_cloud - 1, :] = 0

            # Move the no-longer-needed watermarks to the end of the list
            watermarks = np.roll(watermarks, 1 - watermark_index_above_cloud, axis=0)

            # Edit the new first watermark
            watermarks[0, 0] = cloud_fractional_volume
            watermarks[0, 1:] = 0

        # If none will be overwritten
        else:
            # Move an empty watermark to the start of the list
            watermarks = np.roll(watermarks, 1, axis=0)

            # Edit the partially overwritten watermark
            watermarks[1, 0] -= cloud_fractional_volume

            # Edit the new first watermark
            watermarks[0, 0] = cloud_fractional_volume
            watermarks[0, 1:] = 0

        return watermarks

    def updated_watermarks_from_capture_not_enough(
        self, cloud_fractional_volume, watermarks, enough
    ):
        """
        Update the trap watermarks for capturing electrons when not enough are available to fill every trap below the
        cloud height (rare!).

        Like update_trap_wmk_capture(), but instead of setting filled trap fractions to 1, increase them by the enough
        fraction towards 1.

        Parameters
        ----------
        cloud_fractional_volume : float
            The fractional height of the electron cloud in the pixel.
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        enough : float
            The ratio of available electrons to traps up to this height.

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """

        # Find the highest active watermark
        max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1
        if max_watermark_index == -1:
            max_watermark_index = 0

        # If the first capture
        if max_watermark_index == 0:
            # Edit the new watermark
            watermarks[0, 0] = cloud_fractional_volume
            watermarks[0, 1:] = [
                trap.time_elapsed_from_fill_fraction(enough) for trap in self.traps
            ]

            return watermarks

        # Cumulative watermark heights
        cumulative_watermark_height = np.cumsum(
            watermarks[: max_watermark_index + 1, 0]
        )

        # Find the first watermark above the cloud, which won't be fully overwritten
        watermark_index_above_cloud = np.argmax(
            cloud_fractional_volume < cumulative_watermark_height
        )

        # If all watermarks will be overwritten
        if cumulative_watermark_height[-1] < cloud_fractional_volume:
            # Do the same as the only-some case (unlike update_trap_wmk_capture())
            watermark_index_above_cloud = max_watermark_index

        # If some (or all) will be overwritten
        if 0 < watermark_index_above_cloud:
            # Move one new empty watermark to the start of the list
            watermarks = np.roll(watermarks, 1, axis=0)

            # Reorder the relevant watermarks near the start of the list
            watermarks[: watermark_index_above_cloud + 1] = watermarks[
                1 : watermark_index_above_cloud + 2
            ]

            # Edit the new watermarks' times to correspond to the original fill
            # fraction plus (1 - original fill) * enough.
            # e.g. enough = 0.5 --> time equivalent of fill half way to full.
            # Awkwardly need to convert to fill fractions first using each
            # trap's lifetime separately, but can still do all levels at once.
            ## Actually can't do more than one at a time with continuum traps!
            # watermarks[: watermark_index_above_cloud + 1, 1:] = np.transpose([
            #     trap.time_elapsed_from_fill_fraction(
            #         trap.fill_fraction_from_time_elapsed(time_elapsed) * (1 - enough) + enough
            #     )
            #     for trap, time_elapsed in zip(
            #         self.traps, watermarks[: watermark_index_above_cloud + 1, 1:].T
            #     )
            # ])

            # Awkwardly need to convert to fill fractions first using each
            # trap's lifetime separately.
            for watermark_index in range(watermark_index_above_cloud + 1):
                watermarks[watermark_index, 1:] = [
                    trap.time_elapsed_from_fill_fraction(
                        trap.fill_fraction_from_time_elapsed(time_elapsed)
                        * (1 - enough)
                        + enough
                    )
                    for trap, time_elapsed in zip(
                        self.traps, watermarks[watermark_index, 1:]
                    )
                ]

            # If all watermarks will be overwritten
            if cumulative_watermark_height[-1] < cloud_fractional_volume:
                # Edit the new highest watermark
                watermarks[watermark_index_above_cloud + 1, 0] = (
                    cloud_fractional_volume - cumulative_watermark_height[-1]
                )
                watermarks[watermark_index_above_cloud + 1, 1:] = [
                    trap.time_elapsed_from_fill_fraction(enough) for trap in self.traps
                ]
            else:
                # Edit the new watermarks' heights
                watermarks[
                    watermark_index_above_cloud : watermark_index_above_cloud + 2, 0,
                ] *= (1 - enough)

        # If none will be overwritten
        else:
            # Move an empty watermark to the start of the list
            watermarks = np.roll(watermarks, 1, axis=0)

            # Edit the partially overwritten watermark
            watermarks[1, 0] -= cloud_fractional_volume

            # Edit the new first watermark
            watermarks[0, 0] = cloud_fractional_volume
            watermarks[0, 1:] = [
                trap.time_elapsed_from_fill_fraction(
                    trap.fill_fraction_from_time_elapsed(time_elapsed) * (1 - enough)
                    + enough
                )
                for trap, time_elapsed in zip(self.traps, watermarks[1, 1:])
            ]

        return watermarks
