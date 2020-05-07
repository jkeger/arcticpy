import numpy as np
from scipy import integrate, optimize
from copy import deepcopy
from arctic.ccd import CCD
from arctic import util


class Trap(object):
    def __init__(
        self,
        density=0.13,
        release_timescale=0.25,
        capture_timescale=0,
        ccd=None,
        surface=False,
    ):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        release_timescale : float
            The release timescale of the trap, in the same units as the time 
            spent in each pixel or phase (Clocker sequence).
        capture_timescale : float
            The capture timescale of the trap. Default 0 for instant capture.
            
        Attributes
        ----------
        capture_rate, emission_rate : float
            The capture and emission rates (Lindegren (1998) section 3.2).
        """

        if ccd is None:
            ccd = CCD()

        self.density = density
        self.release_timescale = release_timescale
        self.capture_timescale = capture_timescale
        self.ccd = ccd
        self.surface = surface

        # Rates
        if self.capture_timescale == 0:
            self.capture_rate = np.inf
        else:
            self.capture_rate = 1 / self.capture_timescale
        self.emission_rate = 1 / self.release_timescale

    def cumulative_n_traps_from_n_electrons(self, n_electrons):
        """ Calculate the total number of charge traps exposed to a charge cloud
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

        Parameters
        ----------
        n_electrons : float
            The size of a charge cloud in a pixel, in units of the number of 
            electrons.

        Returns
        -------
        n_traps : float
            The number of traps of this species available.
        """

        #
        # RJM: this is not currently used. But it could be....
        #

        well_depth = self.ccd.full_well_depth
        if self.surface:
            alpha = self.ccd.blooming_level
            beta = 1
            # Let surface traps soak up everything they can, as a cheap way of
            # ensuring that (at least with instantaneous trapping), no pixel in
            # an output image will ever contain more electrons than the full
            # well depth.
            extra_traps = min(n_electrons - well_depth, 0)
        else:
            alpha = self.ccd.well_notch_depth
            beta = self.ccd.well_fill_power
            extra_traps = 0

        n_electrons_available = n_electrons - alpha
        n_traps = (
            self.density
            * util.set_min_max((n_electrons_available) / (well_depth - alpha), 0, 1)
            ** beta
        )
        n_traps += extra_traps

        # Make sure that the effective number of traps available cannot exceed
        # the number of electrons. Adjusting this here is algorithmically much
        # easier than catching lots of excpetions when there are insufficient
        # electrons to fill traps during the capture process.
        n_traps = min(n_traps, n_electrons_available)

        return n_traps

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """ Calculate the fraction of filled traps after a certain time_elapsed.

        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same 
            units as the trap timescales.

        Returns
        -------
        fill : float
            The fraction of filled traps.
        """
        return np.exp(-time_elapsed / self.release_timescale)

    def time_elapsed_from_fill_fraction(self, fill):
        """ Calculate the total time elapsed from the fraction of filled traps.

        Parameters
        ----------
        fill : float
            The fraction of filled traps.

        Returns
        -------
        time_elapsed : float
            The time elapsed, in the same units as the trap timescales.
        """
        return -self.release_timescale * np.log(fill)

    def electrons_released_from_electrons_and_dwell_time(self, electrons, dwell_time=1):
        """ Calculate the number of released electrons from the trap.

        Parameters
        ----------
        electrons : float
            The initial number of trapped electrons.
        dwell_time : float
            The time spent in this pixel or phase, in the same units as the 
            trap timescales.

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
            trap release_timescale.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """
        return np.exp(-time_elapsed / self.release_timescale) * (
            1 - np.exp(-dwell_time / self.release_timescale)
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
            + d_a * (np.arctan((np.log(self.release_timescale) - d_p) / d_w))
            + (
                g_a
                * np.exp(
                    -((np.log(self.release_timescale) - g_p) ** 2.0) / (2 * g_w ** 2.0)
                )
            )
        )

    def __repr__(self):
        return "\n".join(
            (
                "Trap Density: {}".format(self.density),
                "Trap Lifetime: {}".format(self.release_timescale),
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
            for i, trap_i in enumerate(trap):
                poisson_trap.append(
                    Trap(
                        density=densities[i], release_timescale=trap_i.release_timescale
                    )
                )

        return poisson_trap


class TrapInstantCapture(Trap):
    """ For the old C++ style release-then-instant-capture algorithm. """

    def __init__(self, density=0.13, release_timescale=0.25, ccd=None, surface=False):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        release_timescale : float
            The release timescale of the trap, in the same units as the time 
            spent in each pixel or phase (Clocker sequence).
        """

        if ccd is None:
            ccd = CCD()

        self.density = density
        self.release_timescale = release_timescale
        self.ccd = ccd
        self.surface = surface


class TrapLifetimeContinuum(TrapInstantCapture):
    """ For a continuum distribution of release lifetimes for the traps.
        Must be used with TrapManagerTrackTime.
    """

    def __init__(
        self,
        density,
        distribution_of_traps_with_lifetime,
        release_timescale_mu=None,
        release_timescale_sigma=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        distribution_of_traps_with_lifetime : func
            The distribution of traps as a function of release_timescale, middle lifetime, 
            and lifetime scale, such that its integral from 0 to infinity = 1.
            e.g. a log-normal probability density function.
        release_timescale_mu : float
            The middle (e.g. mean or median depending on the distribution) 
            release timescale of the traps.
        release_timescale_sigma : float
            The scale of release lifetimes of the traps.
        """
        super(TrapLifetimeContinuum, self).__init__(
            density=density, release_timescale=release_timescale_mu
        )

        self.distribution_of_traps_with_lifetime = distribution_of_traps_with_lifetime
        self.release_timescale_mu = release_timescale_mu
        self.release_timescale_sigma = release_timescale_sigma

    def fill_fraction_from_time_elapsed(self, time_elapsed):
        """ Calculate the fraction of filled traps after a certain time_elapsed.
    
        Parameters
        ----------
        time_elapsed : float
            The total time elapsed since the traps were filled, in the same 
            units as the trap timescales.

        Returns
        -------
        fill : float
            The fraction of filled traps.
        """

        def integrand(release_timescale, time_elapsed, middle, scale):
            return self.distribution_of_traps_with_lifetime(
                release_timescale, middle, scale
            ) * np.exp(-time_elapsed / release_timescale)

        return integrate.quad(
            integrand,
            0,
            np.inf,
            args=(
                time_elapsed,
                self.release_timescale_mu,
                self.release_timescale_sigma,
            ),
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
            The time elapsed, in the same units as the trap timescales.
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
            trap timescales.

        Returns
        -------
        electrons_released : float
            The number of released electrons.
        """

        def integrand(release_timescale, time_elapsed, dwell_time, middle, scale):
            return (
                self.distribution_of_traps_with_lifetime(
                    release_timescale, middle, scale
                )
                * np.exp(-time_elapsed / release_timescale)
                * (1 - np.exp(-dwell_time / release_timescale))
            )

        return integrate.quad(
            integrand,
            0,
            np.inf,
            args=(
                time_elapsed,
                dwell_time,
                self.release_timescale_mu,
                self.release_timescale_sigma,
            ),
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
        self, density, release_timescale_mu=None, release_timescale_sigma=None,
    ):
        """The parameters for a single trap species.

        Parameters
        ---------
        density : float
            The density of the trap species in a pixel.
        release_timescale_mu : float
            The median release timescale of the traps.
        release_timescale_sigma : float
            The scale of release lifetimes of the traps.
        """

        super(TrapLogNormalLifetimeContinuum, self).__init__(
            density=density,
            distribution_of_traps_with_lifetime=self.log_normal_distribution,
            release_timescale_mu=release_timescale_mu,
            release_timescale_sigma=release_timescale_sigma,
        )
