import numpy as np

from arctic import util


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
