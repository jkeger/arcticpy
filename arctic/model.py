import numpy as np

from arctic import util


class ArcticParams(object):
    def __init__(
        self,
        parallel_ccd_volume=None,
        serial_ccd_volume=None,
        parallel_traps=None,
        serial_traps=None,
    ):
        """Sets up the arctic CTI model using parallel and serial parameters specified using a child of the
        ArcticParams.ParallelParams and ArcticParams.SerialParams abstract base classes.

        Parameters
        ----------
        parallel_ccd_volume: CCDVolume
            Class describing the state of the CCD in the parallel direction
        serial_ccd_volume: CCDVolume
            Class describing the state of the CCD in the serial direction
        parallel_traps : [ArcticParams.ParallelParams]
           The parallel parameters for the arctic CTI model
        serial_traps : [ArcticParams.SerialParams]
           The serial parameters for the arctic CTI model
        """
        self.parallel_ccd_volume = parallel_ccd_volume
        self.serial_ccd_volume = serial_ccd_volume
        self.parallel_traps = parallel_traps or []
        self.serial_traps = serial_traps or []

    @property
    def delta_ellipticity(self):
        return sum(
            [trap.delta_ellipticity for trap in self.parallel_traps]
        ) + sum([trap.delta_ellipticity for trap in self.serial_traps])


class CCDVolume(object):
    def __init__(
        self, well_max_height=1000.0, well_notch_depth=1e-9, well_fill_beta=0.58
    ):
        """Abstract base class of the cti model parameters. Parameters associated with the traps are set via a child \
        class.

        Parameters
        ----------
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """
        self.well_max_height = well_max_height
        self.well_notch_depth = well_notch_depth
        self.well_range = well_max_height - well_notch_depth
        self.well_fill_beta = well_fill_beta

    def __repr__(self):
        return "\n".join(
            (
                "Well Notch Depth: {}".format(self.well_notch_depth),
                "Well Fill Beta: {}".format(self.well_fill_beta),
            )
        )

    # TODO : Its good to give full names to parameters - shorthand may make code cleaer for you but for other peple it
    # TODO : Adds confusion!

    # TODO : For short functions its also more readable using the 'from' naming convention so the reader can see
    # TODO : what does in and out the calculation.

    def electron_fractional_height_from_electrons(self, electrons):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """
        electron_fractional_height = (
            (electrons - self.well_notch_depth) / self.well_range
        ) ** self.well_fill_beta

        return util.set_min_max(electron_fractional_height, 0, 1)


class CCDVolumeComplex(CCDVolume):
    def __init__(
        self,
        well_max_height=1000.0,
        well_notch_depth=1e-9,
        well_fill_alpha=1.0,
        well_fill_beta=0.58,
    ):
        """Abstract base class of the cti model parameters. Parameters associated with the traps are set via a child \
        class.

        Parameters
        ----------
        well_notch_depth : float
            The CCD notch depth
        well_fill_alpha : float
            The volume-filling coefficient (alpha) of how an electron cloud fills the volume of a pixel.
        well_fill_beta : float
            The volume-filling power (beta) of how an electron cloud fills the volume of a pixel.
        well_fill_gamma : float
            The volume-filling constant (gamma) of how an electron cloud fills the volume of a pixel.
        """

        super(CCDVolumeComplex, self).__init__(
            well_max_height=well_max_height,
            well_notch_depth=well_notch_depth,
            well_fill_beta=well_fill_beta,
        )

        self.well_fill_alpha = well_fill_alpha

    def electron_fractional_height_from_electrons(self, electrons):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """

        electron_fractional_height = (
            self.well_fill_alpha
            * (
                (electrons - self.well_notch_depth)
                / (self.well_range - self.well_notch_depth)
            )
        ) ** self.well_fill_beta

        return util.set_min_max(electron_fractional_height, 0, 1)


class Trap(object):
    def __init__(self, density=0.13, lifetime=0.25):
        """The CTI model parameters used for parallel clocking, using one trap of trap.

        Parameters
        ----------
        density : float
            The trap density of the trap.
        lifetime : float
            The trap lifetimes of the trap.
        """
        self.density = density
        self.lifetime = lifetime
        self.exponential_factor = 1 - np.exp(-1 / lifetime)

    def electrons_released_from_electrons(self, electrons):
        """ Calculate the number of released electrons from the trap.

            Args:
                electrons : float
                    The initial number of trapped electrons.

            Returns:
                e_rele : float
                    The number of released electrons.
        """
        return electrons * self.exponential_factor

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
            + (
                g_a
                * np.exp(
                    -((np.log(self.lifetime) - g_p) ** 2.0) / (2 * g_w ** 2.0)
                )
            )
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
        -----------
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
                poisson_trap.append(
                    Trap(density=densities[i], lifetime=s.lifetime)
                )

        return poisson_trap
