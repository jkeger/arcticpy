import numpy as np
from copy import deepcopy

from arctic import util
import arctic as ac


# detector=ac.CCD([3,1,45])
# n_electrons=500
# phase=2
#
# # All the following are equivalent
#
# detector.cloud_fractional_volume_from_n_electrons_and_phase(n_electrons,phase)
#
# f=detector.cloud_fractional_volume_from_n_electrons_in_phase(phase)
# f(n_electrons)
#
# p=ac.CCDPhase(detector,phase)
# p.cloud_fractional_volume_from_n_electrons(n_electrons)


class CCD(object):
    def __init__(
        self,
        fraction_of_traps=[1],
        full_well_depth=1000.0,
        well_notch_depth=0.0,
        well_fill_power=0.58,
        well_bloom_level=None,
    ):
        """
        A model of a CCD detector, and how electrons fill the volume inside each pixel/phase.
        By default, only a single phase is modelled within each pixel. To specify a
        multi-phase device (which will need a comparably complicated clocking sequence to
        be defined in readout electronics), specify the fraction of traps as a list. If the
        phases have different sizes, specify each full_well_depth. If the trap density is 
        uniform, the ratio of full_well_depth to fraction_of_traps will be the same in all
        phases. 

        Parameters
        ----------
        fraction_of_traps : float or [float]
            Assuming that traps have uniform density throughout the CCD, for multi-phase 
            clocking, this specifies the physical width of each phase. This is used only to
            distribute the traps between phases. The units do not matter and can be anything 
            from
            microns to light years, or fractions of a pixel. Only the fractional widths are
            ever returned. If this is an array then you can optionally also enter a list of
            different full_well_depth, well_notch_depth, and well_fill_power for each phase.
        full_well_depth : float or [float]
            The maximum number of electrons that can be contained within a pixel/phase.
            For multiphase clocking, if only one value is supplied, that is (by default)
            replicated to all phases. However, different physical widths of phases can be
            set by specifying the full well depth as a list containing different values.
            If the potential in more than one phase is held high during any stage in the 
            clocking cycle, their full well depths are added together.
            This value is indpependent of the fraction of traps allocated to each phase.
        well_notch_depth : float or [float]
            The number of electrons that fit inside a 'notch' at the bottom of a potential
            well, occupying negligible volume and therefore being immune to trapping. These
            electrons still count towards the full well depth. The notch depth can, in 
            principle, vary between phases.
        well_fill_power : float or [float]
            The exponent in a power-law model of the volume occupied by a cloud of electrons.
            This can, in principle, vary between phases.
        well_bloom_level : float or [float]
            Acts similarly to a notch, but for surface traps.
            Default value is full_well_depth - i.e. no blooming is possible.
        """

        # All parameters are returned as a list of length n_phases
        self.fraction_of_traps = fraction_of_traps
        self.full_well_depth = full_well_depth
        self.well_fill_power = well_fill_power
        self.well_notch_depth = well_notch_depth
        self.well_bloom_level = well_bloom_level

    @property
    def fraction_of_traps(self):
        return self._fraction_of_traps

    @fraction_of_traps.setter
    def fraction_of_traps(self, value):
        if isinstance(value, list):
            self._fraction_of_traps = value
        else:
            self._fraction_of_traps = [value]  # Make sure the arrays are arrays
        #Renormalise?
        #self._pixel_width = sum( i for i in self._fraction_of_traps )
        #self._fraction_of_traps = [i / self._pixel_width for i in self._fraction_of_traps]
        self._n_phases = len(self._fraction_of_traps)

    @property
    def n_phases(self):
        return self._n_phases

    # @property
    # def pixel_width(self):
    #    return self._pixel_width

    @property
    def full_well_depth(self):
        return self._full_well_depth

    @full_well_depth.setter
    def full_well_depth(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in full_well_depth"
                )
            self._full_well_depth = value
        else:
            self._full_well_depth = [value] * self.n_phases

    @property
    def well_fill_power(self):
        return self._well_fill_power

    @well_fill_power.setter
    def well_fill_power(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_fill_power"
                )
            self._well_fill_power = value
        else:
            self._well_fill_power = [value] * self.n_phases

    @property
    def well_notch_depth(self):
        return self._well_notch_depth

    @well_notch_depth.setter
    def well_notch_depth(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_notch_depth"
                )
            self._well_notch_depth = value
        else:
            self._well_notch_depth = [value] * self.n_phases

    # @property
    # def well_range(self):
    #    return [
    #        full_well_depth - well_notch_depth
    #        for full_well_depth, well_notch_depth in zip(self._full_well_depth, self._well_notch_depth)
    #    ]

    @property
    def well_bloom_level(self):
        return self._well_bloom_level

    @well_bloom_level.setter
    def well_bloom_level(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(
                    f"Incorrect number of phases ({len(value)}) in well_bloom_level"
                )
            self._well_bloom_level = value
        else:
            if value is None:
                value = self.full_well_depth
            self._well_bloom_level = [value] * self.n_phases

    # Returns a (self-contained) function describing the well-filling model in a single phase
    def cloud_fractional_volume_from_n_electrons_and_phase(
        self, n_electrons, phase=0, surface=False
    ):
        ccd_phase = self.cloud_fractional_volume_from_n_electrons_in_phase(phase)
        return ccd_phase(n_electrons, surface)

    # Returns a (self-contained) function describing the well-filling model in any phase
    def cloud_fractional_volume_from_n_electrons_in_phase(self, phase=0):
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

        def cloud_fractional_volume_from_n_electrons(n_electrons, surface=False):
            """
            Inputs
            ------
            n_electrons : float
                The size of a charge cloud in a pixel, in units of the number of 
                electrons.
           
            Returns
            -------
            volume : float
                The fraction of traps of this species exposed.
            """
            fraction_of_traps = self.fraction_of_traps[phase]
            full_well_depth = self.full_well_depth[phase]
            well_fill_power = self.well_fill_power[phase]
            well_notch_depth = self.well_notch_depth[phase]
            well_bloom_level = self.well_bloom_level[phase]

            if n_electrons == 0:
                return 0

            if surface:
                empty = self.blooming_level[phase]
                beta = 1
            else:
                empty = self.well_notch_depth[phase]
                beta = self.well_fill_power[phase]
            well_range = self.full_well_depth[phase] - empty

            volume = (
                util.set_min_max((n_electrons - empty) / well_range, 0, 1)
            ) ** beta
            
            return volume

        return cloud_fractional_volume_from_n_electrons

    def cumulative_n_traps_from_n_electrons(self, n_electrons):
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


class CCDPhase(object):
    def __init__(self, ccd=CCD(), phase=0):
        """Extract just one phase from the CCD"""

        self.fraction_of_traps = ccd.fraction_of_traps[phase]
        self.full_well_depth = ccd.full_well_depth[phase]
        self.well_fill_power = ccd.well_fill_power[phase]
        self.well_notch_depth = ccd.well_notch_depth[phase]
        self.well_bloom_level = ccd.well_bloom_level[phase]
        self.cloud_fractional_volume_from_n_electrons = ccd.cloud_fractional_volume_from_n_electrons_in_phase(phase)


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
        """

        super(CCDComplex, self).__init__(
            full_well_depth=full_well_depth,
            well_notch_depth=well_notch_depth,
            well_fill_power=well_fill_power,
        )

        # For multi-phase clocking, use duplicate parameters if not provided
        if self.n_phases > 1:
            if not isinstance(well_fill_alpha, list):
                well_fill_alpha = [well_fill_alpha]

            if len(well_fill_alpha) == 1:
                well_fill_alpha *= phases

                assert len(well_fill_alpha) == self.n_phases

        self.well_fill_alpha = well_fill_alpha

    def extract_phase(self, phase):
        """
        Return a copy of this object with only the single parameter values in 
        the requested phase.
            
        phase : int
            The phase to extract.
        """
        if self.n_phases == 1:
            return self
        else:
            copy = super(CCDComplex, self).extract_phase(self, phase)

            copy.well_fill_alpha = self.well_fill_alpha[phase]

            return copy

    def cloud_fractional_volume_from_electrons(self, n_electrons, phase=0):
        """ Calculate the height the electrons reach within a CCD pixel well.
        """

        assert phase == 0, "TBD"

        cloud_fractional_volume = (
            self.well_fill_alpha
            * (
                (n_electrons - self.well_notch_depth)
                / (
                    self.well_range - self.well_notch_depth
                )  # RJM I think that is a bug because notch depth has been subtracted twice
            )
        ) ** self.well_fill_power

        return util.set_min_max(cloud_fractional_volume, 0, 1)
