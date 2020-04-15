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
            
        Attributes
        ----------
        capture_rate, emission_rate : float
            The capture and emission rates (Lindegren (1998) section 3.2).
        """
        super(TrapSlowCapture, self).__init__(density=density, lifetime=lifetime)

        self.capture_timescale = capture_timescale

        self.capture_rate = 1 / self.capture_timescale
        self.emission_rate = 1 / self.lifetime
