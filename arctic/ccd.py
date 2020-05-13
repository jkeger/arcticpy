import numpy as np
from copy import deepcopy

from arctic import util


class CCD(object):
    def __init__(
        self,
        full_well_depth=1000.0,
        well_notch_depth=0.0,
        well_fill_power=0.58,
        phase_fractional_widths=[1],
        integration_phase=0,
        blooming_level=None,
    ):
        """The parameters for how electrons fill the CCD volume.

        Parameters
        ----------
        full_well_depth : float or [float]
            The maximum height of an electron cloud filling the pixel.
        well_notch_depth : float or [float]
            The CCD notch depth.
        well_fill_power : float or [float]
            The volume-filling power (beta) of how an electron cloud fills the 
            volume of a pixel.
        phase_fractional_widths : float or [float]
            The array or single value of the physical fractional_width of each phase as a 
            fraction of the pixel for multi-phase clocking. If an array then
            optionally enter a list of different full_well_depth, 
            well_notch_depth, and well_fill_power for each phase.
        integration_phase : int
            For multi-phase clocking, the initial phase in which the electrons 
            start when the input image is divided into the separate phases.
        """

        # Parse defaults
        if blooming_level is None:
            fractional_blooming_level = 0.95

        # Make sure the arrays are arrays
        if not isinstance(phase_fractional_widths, list):
            phase_fractional_widths = [phase_fractional_widths]
        self.phases = len(phase_fractional_widths)

        # For multi-phase clocking, use duplicate parameters if not provided
        if self.phases > 1:
            if not isinstance(full_well_depth, list):
                full_well_depth = [full_well_depth]
            if not isinstance(well_notch_depth, list):
                well_notch_depth = [well_notch_depth]
            if not isinstance(well_fill_power, list):
                well_fill_power = [well_fill_power]
            if blooming_level is None:
                blooming_level = [
                    i * fractional_blooming_level for i in full_well_depth
                ]
            if not isinstance(blooming_level, list):
                blooming_level = [blooming_level]

            if len(full_well_depth) == 1:
                full_well_depth *= self.phases
            if len(well_notch_depth) == 1:
                well_notch_depth *= self.phases
            if len(well_fill_power) == 1:
                well_fill_power *= self.phases
            if len(blooming_level) == 1:
                blooming_level *= self.phases

            assert len(full_well_depth) == self.phases
            assert len(well_notch_depth) == self.phases
            assert len(well_fill_power) == self.phases
            assert len(blooming_level) == self.phases
        else:
            if blooming_level is None:
                blooming_level = fractional_blooming_level * full_well_depth

        self.full_well_depth = full_well_depth
        self.well_notch_depth = well_notch_depth
        if self.phases > 1:
            self.well_range = [
                max_height - notch_depth
                for max_height, notch_depth in zip(full_well_depth, well_notch_depth)
            ]
        else:
            self.well_range = full_well_depth - well_notch_depth
        self.well_fill_power = well_fill_power
        self.blooming_level = blooming_level
        self.phase_fractional_widths = phase_fractional_widths
        self.integration_phase = integration_phase

    def __repr__(self):
        return "\n".join(
            (
                "Well Notch Depth: {}".format(self.well_notch_depth),
                "Well Fill Beta: {}".format(self.well_fill_power),
            )
        )

    def extract_phase(self, phase):
        """
        Return a copy of this object with only the single parameter values in 
        the requested phase.
            
        phase : int
            The phase to extract.
        """
        if self.phases == 1:
            return self
        else:
            copy = deepcopy(self)

            copy.full_well_depth = copy.full_well_depth[phase]
            copy.well_notch_depth = copy.well_notch_depth[phase]
            copy.well_range = copy.well_range[phase]
            copy.well_fill_power = copy.well_fill_power[phase]

            return copy

    def cloud_fractional_volume_from_electrons(self, electrons, surface=False):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """
        if electrons == 0:
            return 0

        if surface:
            alpha = self.blooming_level
            beta = 1
        else:
            alpha = self.well_notch_depth
            beta = self.well_fill_power

        cloud_fractional_volume = (
            util.set_min_max(
                (electrons - self.well_notch_depth) / self.well_range, 0, 1
            )
        ) ** self.well_fill_power

        return cloud_fractional_volume


class CCDComplex(CCD):
    """ For a more complex esimate of the height reached by an electron cloud.
    """

    def __init__(
        self,
        full_well_depth=1000.0,
        well_notch_depth=0,
        well_fill_alpha=1.0,
        well_fill_power=0.58,
    ):
        """The parameters for how electrons fill the CCD volume.

        Parameters
        ----------        
        full_well_depth : float or [float]
            The maximum height of an electron cloud filling the pixel.
        well_notch_depth : float or [float]
            The CCD notch depth.
        well_fill_alpha : float or [float]
            The volume-filling coefficient (alpha) of how an electron cloud 
            fills the volume of a pixel.
        well_fill_power : float or [float]
            The volume-filling power (beta) of how an electron cloud fills the 
            volume of a pixel.
        phase_fractional_widths : float or [float]
            The array or single value of the physical fractional_width of each phase as a 
            fraction of the pixel for multi-phase clocking. If an array then
            optionally enter a list of different full_well_depth, 
            well_notch_depth, and well_fill_power for each phase.
        integration_phase : int
            For multi-phase clocking, the initial phase in which the electrons 
            start when the input image is divided into the separate phases.
        """

        super(CCDComplex, self).__init__(
            full_well_depth=full_well_depth,
            well_notch_depth=well_notch_depth,
            well_fill_power=well_fill_power,
        )

        # For multi-phase clocking, use duplicate parameters if not provided
        if self.phases > 1:
            if not isinstance(well_fill_alpha, list):
                well_fill_alpha = [well_fill_alpha]

            if len(well_fill_alpha) == 1:
                well_fill_alpha *= phases

                assert len(well_fill_alpha) == self.phases

        self.well_fill_alpha = well_fill_alpha

    def extract_phase(self, phase):
        """
        Return a copy of this object with only the single parameter values in 
        the requested phase.
            
        phase : int
            The phase to extract.
        """
        if self.phases == 1:
            return self
        else:
            copy = super(CCDComplex, self).extract_phase(self, phase)

            copy.well_fill_alpha = self.well_fill_alpha[phase]

            return copy

    def cloud_fractional_volume_from_electrons(self, electrons):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """

        cloud_fractional_volume = (
            self.well_fill_alpha
            * (
                (electrons - self.well_notch_depth)
                / (
                    self.well_range - self.well_notch_depth
                )  # RJM I think that is a bug because nothch depth has been subtracted twice
            )
        ) ** self.well_fill_power

        return util.set_min_max(cloud_fractional_volume, 0, 1)
