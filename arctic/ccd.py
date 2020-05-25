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
        phase_widths=[1],
        integration_phase=0,
        fraction_of_traps_per_phase=None,
        full_well_depth=1000.0,
        well_notch_depth=0.0,
        well_fill_power=0.58,
        well_bloom_level=None,
    ):
        """The parameters for how electrons fill the CCD volume.

        Parameters
        ----------
        phase_widths : float or [float]
            For multi-phase clocking, the physical width of each phase. The units do 
            not matter and can be anything from microns to light years, or fractions of a pixel.
            Only the fractional widths are ever returned. If this is an array then you can
            optionally also enter a list of different full_well_depth, well_notch_depth, 
            and well_fill_power for each phase.
        full_well_depth : float or [float]
            The maximum height of an electron cloud filling the pixel.
        well_notch_depth : float or [float]
            The CCD notch depth.
        well_fill_power : float or [float]
            The volume-filling power (beta) of how an electron cloud fills the 
            volume of a pixel.
        integration_phase : int
            For multi-phase clocking, the initial phase in which the electrons 
            start when the input image is divided into the separate phases.
        """

        # Parse defaults
        self.phase_widths = phase_widths
        self.integration_phase = integration_phase
        
        self.full_well_depth = full_well_depth
        self.well_fill_power = well_fill_power
        self.well_notch_depth = well_notch_depth
        self.well_bloom_level = well_bloom_level # Default is full_well_depth
        
        # Decide how traps will be distributed between phases
        if fraction_of_traps_per_phase is None:
            fraction_of_traps_per_phase = self.phase_fractional_widths
        else:
            assert len(fraction_of_traps_per_phase) == self.n_phases, "Number of elements in fraction_of_traps_per_phase and phase_widths do not agree."
            total = sum( i for i in fraction_of_traps_per_phase )
            fraction_of_traps_per_phase = [i / total for i in fraction_of_traps_per_phase]
        self.fraction_of_traps_per_phase = fraction_of_traps_per_phase


    @property
    def phase_widths(self):
        return self._phase_widths    
    
    @phase_widths.setter
    def phase_widths(self, value):
        if isinstance(value, list):
            self._phase_widths = value
        else:
            self._phase_widths = [value] # Make sure the arrays are arrays
        self._n_phases = len(self._phase_widths)
        self._pixel_width = sum( i for i in self._phase_widths )

    @property
    def pixel_width(self):
        return self._pixel_width    

    @property
    def n_phases(self):
        return self._n_phases    

    @property
    def phase_fractional_widths(self):
        return [i / self._pixel_width for i in self._phase_widths]    

    @property
    def full_well_depth(self):
        return self._full_well_depth    
    
    @full_well_depth.setter
    def full_well_depth(self, value):
        if isinstance(value, list):
            if len(value) != self.n_phases:
                raise ValueError(f'Incorrect number of phases ({len(value)}) in full_well_depth')
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
                raise ValueError(f'Incorrect number of phases ({len(value)}) in well_fill_power')
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
                raise ValueError(f'Incorrect number of phases ({len(value)}) in well_notch_depth')
            self._well_notch_depth = value
        else:
            self._well_notch_depth = [value] * self.n_phases

    #@property
    #def well_range(self):
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
                raise ValueError(f'Incorrect number of phases ({len(value)}) in well_bloom_level')
            self._well_bloom_level = value
        else:
            if value is None: value = self.full_well_depth
            self._well_bloom_level = [value] * self.n_phases

    
    def cloud_fractional_volume_from_n_electrons_and_phase(self, n_electrons, phase=0, surface=False):
        ccd_phase = self.cloud_fractional_volume_from_n_electrons_in_phase(phase)
        return ccd_phase(n_electrons, surface)


    def cloud_fractional_volume_from_n_electrons_in_phase(self, phase=0):
        def cloud_fractional_volume_from_n_electrons(n_electrons, surface=False):
        
            phase_width = self.phase_widths[phase]
            phase_fractional_width = self.phase_fractional_widths[phase]
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

            volume = phase_fractional_width * (
                util.set_min_max(
                    ( n_electrons - empty ) / well_range, 0, 1
                )
            ) ** beta
            return volume
        
        return cloud_fractional_volume_from_n_electrons



class CCDPhase(object):
    def __init__(self,ccd=CCD(),phase=0):
        """Hello"""
        
        self.phase_widths = ccd.phase_widths[phase]
        self.integration_phase = phase == ccd.integration_phase
        
        self.full_well_depth = ccd.full_well_depth[phase]
        self.well_fill_power = ccd.well_fill_power[phase]
        self.well_notch_depth = ccd.well_notch_depth[phase]
        self.well_bloom_level = ccd.well_bloom_level[phase]

    def cloud_fractional_volume_from_n_electrons(self, n_electrons, surface=False):
    
        phase_width = self.phase_widths
        full_well_depth = self.full_well_depth
        well_fill_power = self.well_fill_power
        well_notch_depth = self.well_notch_depth
        well_bloom_level = self.well_bloom_level  
        
        if n_electrons == 0:
            return 0

        if surface:
            empty = self.blooming_level
            beta = 1
        else:
            empty = self.well_notch_depth
            beta = self.well_fill_power
        well_range = self.full_well_depth - empty

        volume = (
            util.set_min_max(
                ( n_electrons - empty ) / well_range, 0, 1
            )
        ) ** beta
        return volume




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
                )  # RJM I think that is a bug because nothch depth has been subtracted twice
            )
        ) ** self.well_fill_power

        return util.set_min_max(cloud_fractional_volume, 0, 1)


