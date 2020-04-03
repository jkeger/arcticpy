import numpy as np
from scipy import integrate, optimize
from copy import deepcopy


class Trap(object):
    def __init__(self, density=0.13, lifetime=0.25):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        lifetime : float
            The release lifetime of the trap, in the same units as the time 
            spent in each pixel or phase (Clocker sequence).
        """
        self.density = density
        self.lifetime = lifetime

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """ Calculate the fraction of filled traps after a certain time_elapsed.

        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same 
            units as the trap lifetime.

        Returns
        -------
        fill : float
            The fraction of filled traps.
        """
        return np.exp(-time_elapsed / self.lifetime)

    def time_elapsed_from_fill_fraction(self, fill):
        """ Calculate the total time elapsed from the fraction of filled traps.

        Parameters
        ----------
        fill : float
            The fraction of filled traps.

        Returns
        -------
        time_elapsed : float
            The time elapsed, in the same units as the trap lifetime.
        """
        return -self.lifetime * np.log(fill)

    def electrons_released_from_electrons_and_dwell_time(self, electrons, dwell_time=1):
        """ Calculate the number of released electrons from the trap.

        Parameters
        ----------
        electrons : float
            The initial number of trapped electrons.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """
        return electrons * (1 - self.fill_fraction_from_time_elapsed(dwell_time))

    def electrons_released_from_time_elapsed_and_dwell_time(
        self, time_elapsed, dwell_time=1
    ):
        """ Calculate the number of released electrons from the trap.

        Parameters
        ----------
        time_elapsed : float
            The time elapsed since capture.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """
        return np.exp(-time_elapsed / self.lifetime) * (
            1 - np.exp(-dwell_time / self.lifetime)
        )

    @property
    def delta_ellipticity(self):

        a = 0.05333
        d_a = 0.03357
        d_p = 1.628
        d_w = 0.2951
        g_a = 0.09901
        g_p = 0.4553
        g_w = 0.4132

        return self.density * (
            a
            + d_a * (np.arctan((np.log(self.lifetime) - d_p) / d_w))
            + (g_a * np.exp(-((np.log(self.lifetime) - g_p) ** 2.0) / (2 * g_w ** 2.0)))
        )

    def __repr__(self):
        return "\n".join(
            (
                "Trap Density: {}".format(self.density),
                "Trap Lifetime: {}".format(self.lifetime),
            )
        )

    @classmethod
    def poisson_trap(cls, trap, shape, seed=0):
        """For a set of traps with a given set of densities (which are in traps per pixel), compute a new set of \
        trap densities by drawing new values for from a Poisson distribution.

        This requires us to first convert each trap density to the total number of traps in the column.

        This is used to model the random distribution of traps on a CCD, which changes the number of traps in each \
        column.

        Parameters
        ----------
        trap
        shape : (int, int)
            The shape of the image, so that the correct number of trap densities are computed.
        seed : int
            The seed of the Poisson random number generator.
        """
        np.random.seed(seed)
        total_trapss = tuple(map(lambda sp: sp.density * shape[0], trap))
        poisson_densities = [
            np.random.poisson(total_trapss) / shape[0] for _ in range(shape[1])
        ]
        poisson_trap = []
        for densities in poisson_densities:
            for i, s in enumerate(trap):
                poisson_trap.append(Trap(density=densities[i], lifetime=s.lifetime))

        return poisson_trap


class TrapNonUniformHeightDistribution(Trap):
    """ For a non-uniform distribution of traps with height within the pixel.
    """

    def __init__(
        self,
        density,
        lifetime,
        electron_fractional_height_min,
        electron_fractional_height_max,
    ):
        """The parameters for a single trap species. 

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        lifetime : float
            The release lifetime of the trap.
        electron_fractional_height_min, electron_fractional_height_max : float
            The minimum (maximum) fractional height of the electron cloud in 
            the pixel below (above) which corresponds to an effective fractional 
            height of 0 (1), with a linear relation in between.
        """
        super(TrapNonUniformHeightDistribution, self).__init__(
            density=density, lifetime=lifetime
        )

        self.electron_fractional_height_min = electron_fractional_height_min
        self.electron_fractional_height_max = electron_fractional_height_max


class TrapLifetimeContinuum(Trap):
    """ For a continuum distribution of release lifetimes for the traps.
        Must be used with TrapManagerTrackTime.
    """

    def __init__(
        self,
        density,
        distribution_of_traps_with_lifetime,
        middle_lifetime=None,
        scale_lifetime=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        distribution_of_traps_with_lifetime : func
            The distribution of traps as a function of lifetime, middle lifetime, 
            and lifetime scale, such that its integral from 0 to infinity = 1.
            e.g. a log-normal probability density function.
        middle_lifetime : float
            The middle (e.g. mean or median depending on the distribution) 
            release lifetime of the traps.
        scale_lifetime : float
            The scale of release lifetimes of the traps.
        """
        super(TrapLifetimeContinuum, self).__init__(
            density=density, lifetime=middle_lifetime
        )

        self.distribution_of_traps_with_lifetime = distribution_of_traps_with_lifetime
        self.middle_lifetime = middle_lifetime
        self.scale_lifetime = scale_lifetime

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """ Calculate the fraction of filled traps after a certain time_elapsed.
    
        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same 
            units as the trap lifetime.

        Returns
        -------
        fill : float
            The fraction of filled traps.
        """

        def integrand(lifetime, time_elapsed, middle, scale):
            return self.distribution_of_traps_with_lifetime(
                lifetime, middle, scale
            ) * np.exp(-time_elapsed / lifetime)

        return integrate.quad(
            integrand,
            0,
            np.inf,
            args=(time_elapsed, self.middle_lifetime, self.scale_lifetime),
        )[0]

    def time_elapsed_from_fill_fraction(self, fill):
        """ Calculate the total time elapsed from the fraction of filled traps.
    
        Parameters
        ----------
        fill : float
            The fraction of filled traps.

        Returns
        -------
        time_elapsed : float
            The time elapsed, in the same units as the trap lifetime.
        """
        # Crudely iterate to find the time that gives the required fill fraction
        def find_time(time_elapsed):
            return self.fill_fraction_from_time_elapsed(time_elapsed) - fill

        return optimize.fsolve(find_time, 1)[0]

    def electrons_released_from_time_elapsed_and_dwell_time(
        self, time_elapsed, dwell_time=1
    ):
        """ Calculate the number of released electrons from the trap.

        Parameters
        ----------
        time_elapsed : float
            The time elapsed since capture.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap lifetime.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """

        def integrand(lifetime, time_elapsed, dwell_time, middle, scale):
            return (
                self.distribution_of_traps_with_lifetime(lifetime, middle, scale)
                * np.exp(-time_elapsed / lifetime)
                * (1 - np.exp(-dwell_time / lifetime))
            )

        return integrate.quad(
            integrand,
            0,
            np.inf,
            args=(time_elapsed, dwell_time, self.middle_lifetime, self.scale_lifetime),
        )[0]


class TrapLogNormalLifetimeContinuum(TrapLifetimeContinuum):
    """ For a log-normal continuum distribution of release lifetimes for the 
        traps. Must be used with TrapManagerTrackTime.
    """

    @staticmethod
    def log_normal_distribution(x, median, scale):
        """ Return the log-normal probability density.
            
        Parameters
        ----------
        x : float 
            The input value.
        median : float 
            The median of the distribution.
        scale : float 
            The scale of the distribution.
            
        Returns
        --------
        """
        return np.exp(-((np.log(x) - np.log(median)) ** 2) / (2 * scale ** 2)) / (
            x * scale * np.sqrt(2 * np.pi)
        )

    def __init__(
        self, density, middle_lifetime=None, scale_lifetime=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ---------
        density : float
            The density of the trap species in a pixel.
        middle_lifetime : float
            The median release lifetime of the traps.
        scale_lifetime : float
            The scale of release lifetimes of the traps.
        """

        super(TrapLogNormalLifetimeContinuum, self).__init__(
            density=density,
            distribution_of_traps_with_lifetime=self.log_normal_distribution,
            middle_lifetime=middle_lifetime,
            scale_lifetime=scale_lifetime,
        )


class TrapSlowCapture(Trap):
    """ For non-instant capture of electrons combined with their release. """

    def __init__(
        self, density, lifetime, capture_timescale,
    ):
        """The parameters for a single trap species. 

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        lifetime : float
            The release lifetime of the trap.
        capture_timescale : float
            The capture timescale of the trap.
        """
        super(TrapSlowCapture, self).__init__(density=density, lifetime=lifetime)

        self.capture_timescale = capture_timescale

        # Derived parameters for release and capture equations
        self.r_e = 1 / self.lifetime
        self.r_c = 1 / self.capture_timescale


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
        """
        self.traps = traps
        self.rows = rows

        # Set up the watermarks
        self.watermarks = self.initial_watermarks_from_rows_and_total_traps(
            rows=self.rows, total_traps=len(self.traps)
        )

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

    def reset_traps_for_next_express_loop(self):
        """Reset the trap watermarks for the next run of release and capture.
        """
        self.watermarks.fill(0)

    def electrons_released_in_pixel(self, dwell_time=1, width=1):
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
            watermarks = np.roll(watermarks, 1 - watermark_index_above_cloud, axis=0)

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
            watermarks[: 2 * watermark_index_above_cloud] = watermarks[
                1 : 2 * watermark_index_above_cloud + 1
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

    def electrons_captured_in_pixel(self, electrons_available, ccd_volume, width=1):
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

        # Find the number of electrons that should be captured
        electrons_captured = self.electrons_captured_by_traps(
            electron_fractional_height=electron_fractional_height,
            watermarks=self.watermarks,
            traps=self.traps,
            width=width,
        )

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

    def electrons_released_and_captured_in_pixel(
        self, electrons_available, ccd_volume, dwell_time=1, width=1
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
        # Release
        electrons_released = self.electrons_released_in_pixel(
            dwell_time=dwell_time, width=width
        )

        # Capture
        electrons_available += electrons_released
        electrons_captured = self.electrons_captured_in_pixel(
            electrons_available, ccd_volume, width=width
        )

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
        if electron_fractional_height <= self.electron_fractional_height_min:
            return 0
        elif self.electron_fractional_height_max <= electron_fractional_height:
            return 1
        else:
            return (
                electron_fractional_height - self.electron_fractional_height_min
            ) / (
                self.electron_fractional_height_max
                - self.electron_fractional_height_min
            )

    def electrons_captured_in_pixel(self, electrons_available, ccd_volume, width=1):
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

    def electrons_released_in_pixel(self, dwell_time=1, width=1):
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
            watermarks[: 2 * watermark_index_above_cloud] = watermarks[
                1 : 2 * watermark_index_above_cloud + 1
            ]

            # Edit the new watermarks' times to correspond to the original fill
            # fraction plus (1 - original fill) * enough.
            # e.g. enough = 0.5 --> time equivalent of fill half way to full.
            # Awkwardly need to convert to fill fractions first using each
            # trap's lifetime separately, but can still do all levels at once.
            ### Actually can't do more than one at a time with continuum traps!
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

    def electrons_released_and_captured_in_pixel(
        self, electrons_available, ccd_volume, dwell_time=1, width=1
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

        # Find the highest active watermark
        max_watermark_index = np.argmax(self.watermarks[:, 0] == 0) - 1

        # The fractional height the electron cloud reaches in the pixel well
        electron_fractional_height = ccd_volume.electron_fractional_height_from_electrons(
            electrons=electrons_available
        )

        # Find the first watermark above the cloud
        if np.sum(self.watermarks[:, 0]) < electron_fractional_height:
            watermark_index_above_cloud = max_watermark_index + 1
        elif electron_fractional_height == 0:
            watermark_index_above_cloud = -1
        else:
            watermark_index_above_cloud = np.argmax(
                electron_fractional_height < np.cumsum(self.watermarks[:, 0])
            )

        # The number of traps and the rates (Lindegren, 1998) for each species
        densities = np.array([trap.density for trap in self.traps]) * width
        r_c = np.array([trap.r_c for trap in self.traps])
        r_e = np.array([trap.r_e for trap in self.traps])
        r_tot = r_c + r_e

        # Initial number of electrons in traps
        trapped_electrons_initial = np.sum(
            (self.watermarks[:, 0] * self.watermarks[:, 1:].T).T * densities
        )

        # Initialise cumulative watermark height
        cumulative_watermark_height = 0

        # Update all the watermarks for release and/or capture of electrons
        watermark_index = 0
        while watermark_index <= max(max_watermark_index, watermark_index_above_cloud):

            # Update cumulative watermark height
            watermark_height = self.watermarks[watermark_index, 0]
            cumulative_watermark_height += watermark_height

            # Common factor for capture and release probabilities
            exponential_factor = (1 - np.exp(-r_tot * dwell_time)) / r_tot

            # New fill fraction for empty traps
            fill_fraction_from_empty = r_c * exponential_factor

            # New fill fraction for filled traps
            fill_fraction_from_full = 1 - r_e * exponential_factor

            # New fill fraction from only release
            fill_fraction_from_release = 1 - np.exp(-r_e * dwell_time)

            # Initial fill fractions
            fill_fractions_old = deepcopy(self.watermarks[watermark_index, 1:])

            # Release and capture electrons all the way to watermarks below the cloud
            if watermark_index < watermark_index_above_cloud:
                # Update this watermark's fill fractions
                # Release and capture
                self.watermarks[watermark_index, 1:] = (
                    fill_fractions_old * fill_fraction_from_full
                    + (1 - fill_fractions_old) * fill_fraction_from_empty
                )

            # Release and capture electrons part-way to the first existing watermark above the cloud
            elif (
                watermark_index == watermark_index_above_cloud
                and watermark_index_above_cloud <= max_watermark_index
            ):
                # Move one new empty watermark to the start of the list
                self.watermarks = np.roll(self.watermarks, 1, axis=0)

                # Re-set the relevant watermarks near the start of the list
                if watermark_index_above_cloud == 0:
                    self.watermarks[0] = self.watermarks[1]
                else:
                    self.watermarks[
                        : 2 * watermark_index_above_cloud
                    ] = self.watermarks[1 : 2 * watermark_index_above_cloud + 1]

                # Update this and the new split watermarks' fill fractions
                # Release and capture below the cloud
                self.watermarks[watermark_index, 1:] = (
                    fill_fractions_old * fill_fraction_from_full
                    + (1 - fill_fractions_old) * fill_fraction_from_empty
                )
                # Only release, no capture above the cloud
                self.watermarks[watermark_index + 1, 1:] = (
                    fill_fractions_old * fill_fraction_from_release
                )

                # Update this and the new split watermarks' heights
                old_height = self.watermarks[watermark_index, 0]
                self.watermarks[watermark_index, 0] = electron_fractional_height - (
                    cumulative_watermark_height - watermark_height
                )
                self.watermarks[watermark_index + 1, 0] = (
                    old_height - self.watermarks[watermark_index, 0]
                )

                # Increment the index now that an extra watermark has been set
                watermark_index += 1
                max_watermark_index += 1

            # Capture electrons above the highest existing watermark
            elif max_watermark_index < watermark_index_above_cloud:
                # Update this new watermark's fill fractions
                # Only capture, no release
                self.watermarks[watermark_index, 1:] = fill_fraction_from_empty

                # Update the new watermark's height
                self.watermarks[watermark_index, 0] = (
                    electron_fractional_height - cumulative_watermark_height
                )

            # Release electrons from existing watermarks above the cloud
            else:
                # Update this watermark's fill fractions
                # Only release, no capture
                self.watermarks[watermark_index, 1:] = (
                    fill_fractions_old * fill_fraction_from_release
                )

            # Next watermark
            watermark_index += 1

        # Final number of electrons in traps
        trapped_electrons_final = np.sum(
            (self.watermarks[:, 0] * self.watermarks[:, 1:].T).T * densities
        )

        return trapped_electrons_initial - trapped_electrons_final
