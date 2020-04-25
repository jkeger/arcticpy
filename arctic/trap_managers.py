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
        """
        self.traps = traps
        self.rows = rows

        # Set up the watermarks
        self.watermarks = self.initial_watermarks_from_rows_and_total_traps(
            rows=self.rows, total_traps=len(self.traps)
        )

        # Trap densities
        try:
            self.densities = np.array([trap.density for trap in self.traps])
        except AttributeError:
            self.densities = None

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
        return np.zeros((rows, 1 + total_traps), dtype=float)

    def number_of_trapped_electrons_from_watermarks(self, watermarks, width=1):
        """ Sum the total number of electrons currently held in traps.

        Parameters
        ----------
        watermarks : np.ndarray
            The watermarks. See initial_watermarks_from_rows_and_total_traps().
        wdith : float
            The width of this pixel or phase, as a fraction of the whole pixel.
        """
        return (
            np.sum((watermarks[:, 0] * watermarks[:, 1:].T).T * self.densities) * width
        )

    def empty_all_traps(self):
        """Reset the trap watermarks for the next run of release and capture.
        """
        self.watermarks.fill(0)

    def electrons_released_in_pixel(self, dwell_time=1, width=1, express_multiplier=1):
        """ Release electrons from traps and update the trap watermarks.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.
        width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        electrons_released : float
            The number of released electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initialise the number of released electrons
        electrons_released = 0

        # Find the highest active watermark
        #
        # RJM: This is probably unnecessarily slow. Just store the max_watermark_index as a separate variable?
        #      It also has potential problems by comparing a float to zero. After resetting, some heights are only zero within floating point precision
        #      NB: If the code structure is changed, this counter will also need resetting in routines like empty_all_traps()
        #
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1
        # print("max_watermark_index", max_watermark_index)

        # For each watermark
        ## Could do these all at once!
        for watermark_index in range(max_watermark_index + 1):
            # Initialise the number of released electrons from this watermark level
            electrons_released_watermark = 0

            # For each trap species
            for trap_index, trap in enumerate(self.traps):
                # Number of released electrons (not yet including the trap density)
                electrons_released_from_trap = trap.electrons_released_from_electrons_and_dwell_time(
                    electrons=self.watermarks[watermark_index, 1 + trap_index],
                    dwell_time=dwell_time,
                )

                # Update the watermark fill fraction
                self.watermarks[
                    watermark_index, 1 + trap_index
                ] -= electrons_released_from_trap

                # Update the actual number of released electrons
                electrons_released_watermark += (
                    electrons_released_from_trap * trap.density * width
                )

            # Multiply the summed fill fractions by the height
            electrons_released += (
                electrons_released_watermark * self.watermarks[watermark_index, 0]
            )

        return electrons_released

    def electrons_captured_by_traps(
        self, electron_fractional_height, watermarks, traps, width=1
    ):
        """
        Find the total number of electrons that the traps can capture.

        Parameters
        -----------
        electron_fractional_height : float
            The fractional height of the electron cloud in the pixel.
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        traps : [TrapSpecies]
            An array of one or more objects describing a trap of trap.
        width : float
            The width of this pixel or phase, as a fraction of the whole pixel.

        Returns
        -------
        electrons_captured : float
            The number of captured electrons.
        """
        # Initialise the number of captured electrons
        electrons_captured = 0

        # The number of traps of each species
        densities = self.densities * width

        # Find the highest active watermark
        max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1

        # Initialise cumulative watermark height
        cumulative_watermark_height = 0

        # Capture electrons above each existing watermark level below the highest
        for watermark_index in range(max_watermark_index + 1):
            # Update cumulative watermark height
            watermark_height = watermarks[watermark_index, 0]
            cumulative_watermark_height += watermark_height

            # Capture electrons all the way to this watermark (for all traps)
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

    def updated_watermarks_from_capture(self, electron_fractional_height, watermarks):
        """ Update the trap watermarks for capturing electrons.

        Parameters
        ----------
        electron_fractional_height : float
            The fractional height of the electron cloud in the pixel.

        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().

        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """

        # print("updated_watermarks_from_capture")

        # Find the highest active watermark
        max_watermark_index = np.argmax(watermarks[:, 0] == 0) - 1
        if max_watermark_index == -1:
            max_watermark_index = 0

        # Cumulative watermark heights
        cumulative_watermark_height = np.cumsum(
            watermarks[: max_watermark_index + 1, 0]
        )

        # If all watermarks will be overwritten
        if (
            cumulative_watermark_height[-1] < electron_fractional_height
        ):  # RJM can this be <= ? Indeed, should it be, to prevent creating new watermarks in an image with constant flux level?

            # print("A", electron_fractional_height)

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

            # print("B", electron_fractional_height)

            # Edit the partially overwritten watermark
            watermarks[watermark_index_above_cloud, 0] -= (
                electron_fractional_height
                - cumulative_watermark_height[watermark_index_above_cloud - 1]
            )

            # Remove the no-longer-needed overwritten watermarks
            watermarks[: watermark_index_above_cloud - 1, :] = 0

            # Move the no-longer-needed watermarks to the end of the list
            watermarks = np.roll(watermarks, 1 - watermark_index_above_cloud, axis=0)

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = 1

        # If none will be overwritten
        else:

            # print("C", electron_fractional_height)
            # print(watermarks)

            # Move an empty watermark to the start of the list
            watermarks = np.roll(watermarks, 1, axis=0)

            # Edit the partially overwritten watermark
            watermarks[1, 0] -= electron_fractional_height

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = 1

        return watermarks

    def updated_watermarks_from_capture_not_enough(
        self, electron_fractional_height, watermarks, enough
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
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = enough

            return watermarks

        # Cumulative watermark heights
        cumulative_watermark_height = np.cumsum(
            watermarks[: max_watermark_index + 1, 0]
        )

        # Find the first watermark above the cloud, which won't be fully overwritten
        watermark_index_above_cloud = np.argmax(
            electron_fractional_height < cumulative_watermark_height
        )

        # If all watermarks will be overwritten
        if cumulative_watermark_height[-1] < electron_fractional_height:
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

            # Edit the new watermarks' fill fractions to the original fill plus (1 -
            # original fill) * enough.
            # e.g. enough = 0.5 --> fill half way to 1.
            watermarks[: watermark_index_above_cloud + 1, 1:] = (
                watermarks[: watermark_index_above_cloud + 1, 1:] * (1 - enough)
                + enough
            )

            # If all watermarks will be overwritten
            if cumulative_watermark_height[-1] < electron_fractional_height:
                # Edit the new highest watermark
                watermarks[watermark_index_above_cloud + 1, 0] = (
                    electron_fractional_height - cumulative_watermark_height[-1]
                )
                watermarks[watermark_index_above_cloud + 1, 1:] = enough
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
            watermarks[1, 0] -= electron_fractional_height

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = watermarks[1, 1:] * (1 - enough) + enough

        return watermarks

    def fraction_of_exposed_traps(self, n_free_electrons, ccd_volume):
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
            n_exposed_traps = max(n_free_electrons - ccd_volume.blooming_level, 0)
            # RJM: what density to use if more than one trap species?
            fraction_of_exposed_traps = min(n_exposed_traps / self.densities, 1)
            if 0 < n_exposed_traps < 1:
                print(
                    "hello",
                    n_free_electrons,
                    ccd_volume.blooming_level,
                    fraction_of_exposed_traps,
                )
            # fraction_of_exposed_traps = n_exposed_traps / self.densities

        else:
            fraction_of_exposed_traps = ccd_volume.electron_fractional_height_from_electrons(
                electrons=n_free_electrons
            )

        return fraction_of_exposed_traps

    def electrons_captured_in_pixel(
        self, n_free_electrons, ccd_volume, width=1, express_multiplier=1
    ):
        """
        Considering all of the charge traps in a pixel, 
          (a) determine how many electrons will be captured
          (b) update the trap watermarks and occupancy levels to reflect that 
              number of captured electrons.
        Both of these steps involve a loop over all the traps in a pixel.
        Unfortunately, the two loops cannot be combined in case insufficient
        electrons are available to fill all the exposed traps (this happens
        particularly in bias images or with low values of express that amplify
        the effect of each capture event). In that case, the number of captured
        electrons is reduced, and the watermarks are adjusted so that each trap
        captures an (identical) fraction of the electrons that it could have.

        Parameters
        ----------
        n_free_electrons : float
            The number of available electrons for trapping.
        ccd_volume : CCDVolume
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCDVolume.extract_phase().
        width : float
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

        # Zero capture if there are no free electrons in the pixel
        if n_free_electrons <= 0:
            return 0

        # The fractional height the electron cloud reaches in the pixel well
        # print(electrons_available)
        # fraction_of_exposed_traps_old = self.fractional_area_of_charge_cloud(
        #    electrons_available, ccd_volume
        # )
        # fraction_of_exposed_traps_old = ccd_volume.electron_fractional_height_from_electrons(
        #   electrons=electrons_available
        # )

        fraction_of_exposed_traps = self.fraction_of_exposed_traps(
            n_free_electrons, ccd_volume
        )
        # assert fraction_of_exposed_traps == fraction_of_exposed_traps_old

        # Modify the effective height for non-uniform trap distributions
        #
        # RJM: this is the only thing different in this entire class! Can duplicate code be excised
        #      by creating a method self.electron_fractional_height_from_electrons in Trap, which
        #      just calls ccd_volume.electron_fractional_height_from_electrons, but modifying it here?
        #
        # electron_fractional_height = self.effective_non_uniform_electron_fractional_height(
        #    electron_fractional_height
        # )
        # print("hello",fraction_of_exposed_traps,fraction_of_exposed_traps)
        # fraction_of_exposed_traps = ccd_volume.electron_fractional_height_from_electrons(
        #    electrons=electrons_available
        # )
        # print("hello1",self.densities)
        # for trap in self.traps:
        #    fraction_of_exposed_traps = trap.cumulative_n_traps_from_n_electrons(
        #       electrons_available) / trap.density
        #    print("hello2",fraction_of_exposed_traps)
        # print("hello3",fraction_of_exposed_traps)

        # Zero capture if no electrons are high enough to be trapped
        if fraction_of_exposed_traps <= 0:
            return 0

        #
        # RJM: the next check should NOT be needed! But unit tests fail if it is omitted. I don't understand...
        #
        # Zero capture if no electrons are high enough to be trapped
        if n_free_electrons < ccd_volume.well_notch_depth:
            return 0

        # Find the number of electrons that should be captured
        n_electrons_captured = self.electrons_captured_by_traps(
            electron_fractional_height=fraction_of_exposed_traps,
            watermarks=self.watermarks,
            traps=self.traps,
            width=width,
        )

        # Update the watermarks to reflect new occupancy levels
        n_electrons_required = n_electrons_captured * express_multiplier
        if n_electrons_required <= 0:
            # If no electrons are captured, don't need to do anything
            return 0
        elif n_electrons_required <= n_free_electrons:
            # Update watermarks as expected
            self.watermarks = self.updated_watermarks_from_capture(
                electron_fractional_height=fraction_of_exposed_traps,
                watermarks=self.watermarks,
            )
            return n_electrons_captured
        else:
            # Update watermarks so that each trap captures a fraction of the
            # electrons it would have been able to.
            # NB: in the C++ version of arCTIc, this was implemented by
            # filling traps completely but returning an adjusted value of
            # express_multiplier. The python model is arguably more realistic
            # to the underlying physics, and it is algorithmically easier
            # to ensure that the right number of captured electrons are
            # available for release in subsequent pixel-to-pixel transfers.
            fraction_of_required_electrons_available = (
                n_free_electrons / n_electrons_required
            )  # Have checked in first if condition that the denominator is nonzero
            # print("Enough", fraction_of_required_electrons_available)
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                electron_fractional_height=fraction_of_exposed_traps,
                watermarks=self.watermarks,
                enough=fraction_of_required_electrons_available,
            )
            return n_electrons_captured * fraction_of_required_electrons_available

    def electrons_released_and_captured_in_pixel(
        self,
        electrons_available,
        ccd_volume,
        dwell_time=1,
        width=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.

        Parameters
        ----------
        electrons_available : float
            The number of available electrons for trapping.
        ccd_volume : CCDVolume
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCDVolume.extract_phase().
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.
        width : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        net_electrons_released_and_captured : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """

        # print("Processing instantaneous charge capture")
        # print("Initial:",type(electrons_available))

        # Release
        # print(
        #    "Total number of electrons in traps",
        #    np.sum(self.watermarks[:, 0].T * self.watermarks[:, 1]),
        # )
        # print("Release")
        electrons_released = self.electrons_released_in_pixel(
            dwell_time=dwell_time, width=width, express_multiplier=express_multiplier
        )
        electrons_available += electrons_released
        # print("Released:",type(electrons_released))

        # print(self.watermarks[:, 1])
        # print(
        #    "Total number of electrons in traps",
        #    np.sum(self.watermarks[:, 0].T * self.watermarks[:, 1]),
        # )
        # print("Capture")

        # Capture
        electrons_captured = self.electrons_captured_in_pixel(
            electrons_available,
            ccd_volume,
            width=width,
            express_multiplier=express_multiplier,
        )
        # print("Captured:",type(electrons_captured))
        # print(self.watermarks[:, 1])
        # print(
        #    "Total number of electrons in traps",
        #    np.sum(self.watermarks[:, 0].T * self.watermarks[:, 1]),
        # )
        # print("Done")

        return electrons_released - electrons_captured


class TrapManagerNonUniformHeightDistribution(TrapManager):
    """ For a non-uniform distribution of traps with height within the pixel. """

    def __init__(
        self, traps, rows,
    ):
        """The manager for potentially multiple trap species that must use 
        watermarks in the same way as each other.
        
        Allows a non-uniform distribution of trap heights within the pixel, by 
        modifying the effective height that the electron cloud reaches in terms 
        of the proportion of traps that are reached. Must be the same for all 
        traps in this manager.

        Parameters
        ----------
        traps : [Trap]
            A list of one or more trap objects.
        rows :int
            The number of rows in the image. i.e. the maximum number of
            possible electron trap/release events.
        """
        super(TrapManagerNonUniformHeightDistribution, self).__init__(
            traps=traps, rows=rows
        )

        #
        # RJM: why [0] in the next two lines?
        #
        self.electron_fractional_height_min = traps[0].electron_fractional_height_min
        self.electron_fractional_height_max = traps[0].electron_fractional_height_max

    def effective_non_uniform_electron_fractional_height(
        self, electron_fractional_height
    ):
        """
        Find the total number of electrons that the traps can capture, for a 
        non-uniform distribution of traps with height within the pixel.

        Parameters
        -----------
        electron_fractional_height : float
            The original fractional height of the electron cloud in the pixel.
            
        Returns
        -------
        electron_fractional_height : float
            The effective fractional height of the electron cloud in the pixel
            given the distribution of traps with height within the pixel.
        """
        # print("NonUniformHeight",electron_fractional_height,self.electron_fractional_height_min,self.electron_fractional_height_max)
        if electron_fractional_height <= self.electron_fractional_height_min:
            return 0
        elif electron_fractional_height >= self.electron_fractional_height_max:
            return 1
        else:
            return (
                electron_fractional_height - self.electron_fractional_height_min
            ) / (
                self.electron_fractional_height_max
                - self.electron_fractional_height_min
            )

    def electrons_captured_in_pixel(
        self, electrons_available, ccd_volume, width=1, express_multiplier=1
    ):
        """
        Capture electrons in traps and update the trap watermarks.

        Parameters
        ----------
        electrons_available : float
            The number of available electrons for trapping.
        ccd_volume : CCDVolume
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCDVolume.extract_phase().
        width : float
            The width of this pixel or phase, as a fraction of the whole pixel.

        Returns
        -------
        electrons_captured : float
            The number of captured electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """

        # Zero capture if no electrons are high enough to be trapped
        if electrons_available < ccd_volume.well_notch_depth:
            return 0

        # The fractional height the electron cloud reaches in the pixel well
        electron_fractional_height = ccd_volume.electron_fractional_height_from_electrons(
            electrons=electrons_available
        )

        # Modify the effective height for non-uniform trap distributions
        #
        # RJM: this is the only thing different in this entire class! Can duplicate code be excised
        #      by creating a method self.electron_fractional_height_from_electrons in Trap, which
        #      just calls ccd_volume.electron_fractional_height_from_electrons, but modifying it here?
        #
        electron_fractional_height = self.effective_non_uniform_electron_fractional_height(
            electron_fractional_height
        )

        # Find the number of electrons that should be captured
        electrons_captured = self.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=self.watermarks,
            traps=self.traps,
            width=width,
        )

        # print("capturing in surface traps",electrons_captured,electrons_available)

        # Stop if no capture
        if electrons_captured == 0:
            return electrons_captured

        # Check whether enough electrons are available to be captured
        enough = electrons_available / electrons_captured

        # Update watermark levels
        if 1 < enough:
            self.watermarks = self.updated_watermarks_from_capture(
                electron_fractional_height=electron_fractional_height,
                watermarks=self.watermarks,
            )
        else:
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                electron_fractional_height=electron_fractional_height,
                watermarks=self.watermarks,
                enough=enough,
            )
            # Reduce the final number of captured electrons
            electrons_captured *= enough

        return electrons_captured


class TrapManagerTrackTime(TrapManager):
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

    def electrons_released_in_pixel(self, dwell_time=1, width=1, express_multiplier=1):
        """ Release electrons from traps and update the trap watermarks.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.
        wdith : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        electrons_released : float
            The number of released electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # Initialise the number of released electrons
        electrons_released = 0

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # For each watermark
        for watermark_index in range(max_watermark_index + 1):
            # Initialise the number of released electrons from this watermark level
            electrons_released_watermark = 0

            # For each trap species
            for trap_index, trap in enumerate(self.traps):
                # Number of released electrons (not yet including the trap density)
                electrons_released_from_trap = trap.electrons_released_from_time_elapsed_and_dwell_time(
                    time_elapsed=self.watermarks[watermark_index, 1 + trap_index],
                    dwell_time=dwell_time,
                )

                # Update the watermark times
                self.watermarks[watermark_index, 1 + trap_index] += dwell_time

                # Update the actual number of released electrons
                electrons_released_watermark += (
                    electrons_released_from_trap * trap.density * width
                )

            # Multiply the summed fill fractions by the height
            electrons_released += (
                electrons_released_watermark * self.watermarks[watermark_index, 0]
            )

        return electrons_released

    def electrons_captured_by_traps(
        self, electron_fractional_height, watermarks, traps, width=1
    ):
        """
        Find the total number of electrons that the traps can capture.

        Parameters
        -----------
        electron_fractional_height : float
            The fractional height of the electron cloud in the pixel.
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        traps : [TrapSpecies]
            An array of one or more objects describing a trap of trap.
        wdith : float
            The width of this pixel or phase, as a fraction of the whole pixel.

        Returns
        -------
        electrons_captured : float
            The number of captured electrons.
        """
        # Initialise the number of captured electrons
        electrons_captured = 0

        # The number of traps of each species
        densities = np.array([trap.density for trap in traps]) * width

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
            if cumulative_watermark_height < electron_fractional_height:
                electrons_captured += watermark_height * np.sum(
                    (1 - fill_fractions) * densities
                )

            # Capture electrons part-way between the previous and this watermark
            else:
                electrons_captured += (
                    electron_fractional_height
                    - (cumulative_watermark_height - watermark_height)
                ) * np.sum((1 - fill_fractions) * densities)

                # No point in checking even higher watermarks
                return electrons_captured

        # Capture any electrons above the highest existing watermark
        if watermarks[max_watermark_index, 0] < electron_fractional_height:
            electrons_captured += (
                electron_fractional_height - cumulative_watermark_height
            ) * np.sum(densities)

        return electrons_captured

    def updated_watermarks_from_capture(self, electron_fractional_height, watermarks):
        """ Update the trap watermarks for capturing electrons.

        Parameters
        ----------
        electron_fractional_height : float
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
        if cumulative_watermark_height[-1] < electron_fractional_height:
            # Overwrite the first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = 0

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
            watermarks = np.roll(watermarks, 1 - watermark_index_above_cloud, axis=0)

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = 0

        # If none will be overwritten
        else:
            # Move an empty watermark to the start of the list
            watermarks = np.roll(watermarks, 1, axis=0)

            # Edit the partially overwritten watermark
            watermarks[1, 0] -= electron_fractional_height

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = 0

        return watermarks

    def updated_watermarks_from_capture_not_enough(
        self, electron_fractional_height, watermarks, enough
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
            watermarks[0, 0] = electron_fractional_height
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
            electron_fractional_height < cumulative_watermark_height
        )

        # If all watermarks will be overwritten
        if cumulative_watermark_height[-1] < electron_fractional_height:
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
            if cumulative_watermark_height[-1] < electron_fractional_height:
                # Edit the new highest watermark
                watermarks[watermark_index_above_cloud + 1, 0] = (
                    electron_fractional_height - cumulative_watermark_height[-1]
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
            watermarks[1, 0] -= electron_fractional_height

            # Edit the new first watermark
            watermarks[0, 0] = electron_fractional_height
            watermarks[0, 1:] = [
                trap.time_elapsed_from_fill_fraction(
                    trap.fill_fraction_from_time_elapsed(time_elapsed) * (1 - enough)
                    + enough
                )
                for trap, time_elapsed in zip(self.traps, watermarks[1, 1:])
            ]

        return watermarks


class TrapManagerSlowCapture(TrapManager):
    """ For non-instant capture of electrons combined with their release. """

    def __init__(
        self, traps, rows,
    ):
        """The manager for potentially multiple trap species that must use 
        watermarks in the same way as each other.

        Attributes
        ----------
        capture_rates, emission_rates, total_rates : np.ndarray
            The capture, emission, and total rates of all the trap species 
            (Lindegren (1998) section 3.2).
        """
        super(TrapManagerSlowCapture, self).__init__(traps=traps, rows=rows)

        self.capture_rates = np.array([trap.capture_rate for trap in self.traps])
        self.emission_rates = np.array([trap.emission_rate for trap in self.traps])
        self.total_rates = self.capture_rates + self.emission_rates

    def initial_watermarks_from_rows_and_total_traps(self, rows, total_traps):
        # Allow for extra watermarks that may be created by the modified algorithm
        return super(
            TrapManagerSlowCapture, self
        ).initial_watermarks_from_rows_and_total_traps(rows * 2, total_traps)

    def fill_probabilities_from_dwell_time(self, dwell_time):
        """ The probabilities of being full after release and/or capture.
        
        See Lindegren (1998) section 3.2.

        Parameters
        ----------
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.
            
        Returns
        -------
        fill_probability_from_empty : float
            The fraction of traps that were empty that become full.
        fill_probability_from_full : float
            The fraction of traps that were full that stay full.
        fill_probability_from_release : float
            The fraction of traps that were full that stay full after release.
        """
        # Common factor for capture and release probabilities
        exponential_factor = (
            1 - np.exp(-self.total_rates * dwell_time)
        ) / self.total_rates

        # New fill fraction for empty traps (Eqn. 20)
        fill_probability_from_empty = self.capture_rates * exponential_factor

        # New fill fraction for filled traps (Eqn. 21)
        fill_probability_from_full = 1 - self.emission_rates * exponential_factor

        # New fill fraction from only release
        fill_probability_from_release = 1 - np.exp(-self.emission_rates * dwell_time)

        return (
            fill_probability_from_empty,
            fill_probability_from_full,
            fill_probability_from_release,
        )

    def watermark_index_above_cloud_from_electron_fractional_height(
        self, electron_fractional_height, watermarks, max_watermark_index
    ):
        """ Return the index of the first watermark above the cloud.
            
        Parameters
        ----------
        electron_fractional_height : float
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
        if np.sum(watermarks[:, 0]) < electron_fractional_height:
            return max_watermark_index + 1

        elif electron_fractional_height == 0:
            return -1

        else:
            return np.argmax(electron_fractional_height < np.cumsum(watermarks[:, 0]))

    def update_watermark_heights_for_cloud_below_highest(
        self, watermarks, electron_fractional_height, watermark_index_above_cloud
    ):
        """ Update the trap watermarks for a cloud below the highest watermark.
            
        Parameters
        ----------
        watermarks : np.ndarray
            The initial watermarks. See initial_watermarks_from_rows_and_total_traps().
        electron_fractional_height : float
            The fractional height of the electron cloud in the pixel.
        watermark_index_above_cloud : int
            The index of the first watermark above the cloud.
            
        Returns
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """
        # The height and cumulative height of the watermark around the cloud height
        watermark_height = self.watermarks[watermark_index_above_cloud, 0]
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
        old_height = watermarks[watermark_index_above_cloud, 0]
        watermarks[watermark_index_above_cloud, 0] = electron_fractional_height - (
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

    def electrons_released_and_captured_in_pixel(
        self,
        electrons_available,
        ccd_volume,
        dwell_time=1,
        width=1,
        express_multiplier=1,
    ):
        """ Release and capture electrons and update the trap watermarks.
        
        See Lindegren (1998) section 3.2.

        Parameters
        ----------
        electrons_available : float
            The number of available electrons for trapping.
        ccd_volume : CCDVolume
            The object describing the CCD. Must have only a single value for 
            each parameter, as set by CCDVolume.extract_phase().
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.
        wdith : float
            The width of this pixel or phase, as a fraction of the whole pixel.
            
        Returns
        -------
        net_electrons_released_and_captured : float
            The net number of released (if +ve) and captured (if -ve) electrons.
        
        Updates
        -------
        watermarks : np.ndarray
            The updated watermarks. See initial_watermarks_from_rows_and_total_traps().
        """

        # print("Processing slow capture")

        # Initial watermarks and number of electrons in traps
        watermarks_initial = deepcopy(self.watermarks)
        trapped_electrons_initial = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, width=width
        )

        # The number of traps for each species
        densities = self.densities * width

        # Probabilities of being full after release and/or capture
        (
            fill_probability_from_empty,
            fill_probability_from_full,
            fill_probability_from_release,
        ) = self.fill_probabilities_from_dwell_time(dwell_time=dwell_time)

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional height the electron cloud reaches in the pixel well
        electron_fractional_height = ccd_volume.electron_fractional_height_from_electrons(
            electrons=electrons_available
        )

        # Find the first watermark above the cloud
        watermark_index_above_cloud = self.watermark_index_above_cloud_from_electron_fractional_height(
            electron_fractional_height=electron_fractional_height,
            watermarks=self.watermarks,
            max_watermark_index=max_watermark_index,
        )

        # First capture: make the new watermark then can return immediately
        if max_watermark_index == -1:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[0, 0] = electron_fractional_height
            watermarks_initial[0, 0] = self.watermarks[0, 0]

            # Update the fill fractions
            self.watermarks[0, 1:] = fill_probability_from_empty

            # Final number of electrons in traps
            trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, width=width
            )

            # Not enough available electrons to capture
            #
            # RJM The following can cause division by zero errors
            #
            enough = electrons_available / trapped_electrons_final
            if enough < 1:
                # For watermark fill fractions that increased, tweak them such that
                #   the resulting increase instead matches the available electrons
                self.watermarks = self.updated_watermarks_from_capture_not_enough(
                    self.watermarks, watermarks_initial, enough
                )

                # Final number of electrons in traps
                trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                    watermarks=self.watermarks, width=width
                )

            return -trapped_electrons_final

        # Cloud height below existing watermarks: create a new watermark at the
        #   cloud height then release electrons from watermarks above the cloud
        elif (
            watermark_index_above_cloud <= max_watermark_index
            or max_watermark_index == -1
        ):

            # Create the new watermark at the cloud height
            if electron_fractional_height > 0:

                # Update the watermark heights, duplicated for the initial watermarks
                self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=self.watermarks,
                    electron_fractional_height=electron_fractional_height,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )
                watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
                    watermarks=watermarks_initial,
                    electron_fractional_height=electron_fractional_height,
                    watermark_index_above_cloud=watermark_index_above_cloud,
                )

                # Increment the index now that an extra watermark has been set
                max_watermark_index += 1

            # Release electrons from existing watermark levels above the cloud
            # Update the fill fractions
            self.watermarks[watermark_index_above_cloud + 1 :, 1:] = (
                self.watermarks[watermark_index_above_cloud + 1 :, 1:]
                * fill_probability_from_release
            )

            # Current numbers of electrons temporarily in traps and now available
            trapped_electrons_tmp = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, width=width
            )
            electrons_available += trapped_electrons_initial - trapped_electrons_tmp

            # Re-calculate the height of the electron cloud
            electron_fractional_height = ccd_volume.electron_fractional_height_from_electrons(
                electrons=electrons_available
            )
            watermark_index_above_cloud = self.watermark_index_above_cloud_from_electron_fractional_height(
                electron_fractional_height=electron_fractional_height,
                watermarks=self.watermarks,
                max_watermark_index=max_watermark_index,
            )

            # Update the watermark heights, duplicated for the initial watermarks
            self.watermarks = self.update_watermark_heights_for_cloud_below_highest(
                watermarks=self.watermarks,
                electron_fractional_height=electron_fractional_height,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )
            watermarks_initial = self.update_watermark_heights_for_cloud_below_highest(
                watermarks=watermarks_initial,
                electron_fractional_height=electron_fractional_height,
                watermark_index_above_cloud=watermark_index_above_cloud,
            )

            # Increment the index now that an extra watermark has been set
            max_watermark_index += 1

        # Cloud height above existing watermarks: initialise the new watermark
        else:
            # Update the watermark height, duplicated for the initial watermarks
            self.watermarks[
                watermark_index_above_cloud, 0
            ] = electron_fractional_height - np.sum(self.watermarks[:, 0])
            watermarks_initial[watermark_index_above_cloud, 0] = self.watermarks[
                watermark_index_above_cloud, 0
            ]

        # Release and capture electrons all the way to watermarks below the cloud
        fill_fractions_old = self.watermarks[: watermark_index_above_cloud + 1, 1:]
        # print("fill_fractions_old",fill_fractions_old)
        # print("fill_probability_from_full",fill_probability_from_full)
        # print("fill_probability_from_empty",fill_probability_from_empty)
        self.watermarks[: watermark_index_above_cloud + 1, 1:] = (
            fill_fractions_old * fill_probability_from_full
            + (1 - fill_fractions_old) * fill_probability_from_empty
        )

        # Final number of electrons in traps
        trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
            watermarks=self.watermarks, width=width
        )

        # RJM added to prevent division by zero errors
        if trapped_electrons_final == trapped_electrons_initial:
            return 0

        # Not enough available electrons to capture
        enough = electrons_available / (
            trapped_electrons_final - trapped_electrons_initial
        )
        if 0 < enough and enough < 1:
            # For watermark fill fractions that increased, tweak them such that
            #   the resulting increase instead matches the available electrons
            self.watermarks = self.updated_watermarks_from_capture_not_enough(
                self.watermarks, watermarks_initial, enough
            )

            # Final number of electrons in traps
            trapped_electrons_final = self.number_of_trapped_electrons_from_watermarks(
                watermarks=self.watermarks, width=width
            )

        return trapped_electrons_initial - trapped_electrons_final
