""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy

class ROE(object):
    def __init__(
        self,
        dwell_times=[1],
        integration_step=None, # Why should this matter?
        charge_injection=False,
        empty_traps_at_start=True,
        empty_traps_between_columns=True,
    ):
        """
        The parameters for the clocking of electrons with read-out electronics.

        Parameters
        ----------
        sequence : float or [float]
            The array or single value of the time between clock shifts in each 
            phase or single phase. The trap release lifetimes must be in the 
            same units. Different CCDVolume parameters may be used for each 
            phase.
        charge_injection : bool
            False: usual imaging operation of CCD, where (photo)electrons are 
                   created in pixels then clocked through however many pixels lie 
                   between it and the readout register.
            True:  electrons are electronically created by a charge injection 
                   structure at the end of a CCD, then clocked through ALL of the 
                   pixels to the readout register. By default, it is assumed that 
                   this number is the number of rows in the image (but see n_rows).
        empty_traps_at_start : bool
            True:  begin the readout process with empty traps, so some electrons
                   in the input image are immediately lost. 
            False: 
        empty_traps_between_columns : bool
            True:  each column has independent traps (appropriate for parallel 
                   clocking)
            False: each column moves through the same traps, which therefore
                   preserve occupancy, allowing trails to extend onto the next 
                   column (appropriate for serial clocking, if all prescan and
                   overscan pixels are included in the image array).
        integration_step : int
            Which step of the clocking sequence is used to collect photons, i.e.
            during integration or charge injection.
        """
        # The units of dwell times matter!

        #Deprecated
        #self.sequence=[1]
        
        # Check input number of dwell times as an indication of the desired
        # number of steps in the clocking sequence
        if not isinstance(dwell_times, list):
            dwell_times = [dwell_times]
        self.n_steps = len(dwell_times) 
        self.dwell_times = dwell_times  # The units of these matter!

        self.charge_injection = charge_injection
        self.empty_traps_at_start = empty_traps_at_start
        self.empty_traps_between_columns = empty_traps_between_columns
       
        self.clock_sequence = self._generate_clock_sequence(self.n_steps, integration_step=integration_step)
        self.n_phases = len(self.clock_sequence[0])
        self.integration_step = integration_step
        
        # Record which range of charge clouds are accessed during the clocking sequence
        referred_to_pixels=[0]
        for step in range(self.n_steps):
            for phase in range(self.n_phases):
                referred_to_pixels.extend([
                    self.clock_sequence[step][phase]['capture_from_which_pixel'],
                    self.clock_sequence[step][phase]['release_to_which_pixel']
                ])
        self.min_referred_to_pixel = min(referred_to_pixels)
        self.max_referred_to_pixel = max(referred_to_pixels)
        
        
    def _generate_clock_sequence(self, n_steps, integration_step=None):

        """ 
        For a basic readout sequence (involving one step always held high), a list of
        steps describing the readout electronics at each stage of a clocking sequence,
        during a basic readout with a single phase always held high (and instant transfer
        between these phases). 
        """

        # The number of steps need not be the same as the number of phases 
        # in the CCD, but is for a simple scheme.
        n_phases = n_steps
        
        clock_sequence = []
        for step in range(n_steps):
            potentials = []
            for phase in range(n_phases):
                high_phase = step #n_phases - phase
                high = (phase == high_phase)
                adjacent_phases_high = [high_phase]
                capture_from_which_pixel = 0
                n_steps_around_into_same_pixel = n_phases // 2 # 1 for 3 or 4 phase devices
                if phase > ( step + n_steps_around_into_same_pixel):
                    release_to_which_pixel = -1
                elif phase < (step -n_steps_around_into_same_pixel):
                    release_to_which_pixel = 1 # Release into charge cloud following (for normal readout)
                else:
                    release_to_which_pixel = 0 # Release into same charge cloud
                potential={'high':high,
                           'adjacent_phases_high':adjacent_phases_high,
                           'capture_from_which_pixel':capture_from_which_pixel,
                           'release_to_which_pixel':release_to_which_pixel}
                potentials.append(potential)
                #potentials.append(ROEPotential(
                #    high=high,
                #    adjacent_phases_high=adjacent_phases_high,
                #    capture_from_which_pixel=capture_from_which_pixel,
                #    release_to_which_pixel=release_to_which_pixel,
                #))
            clock_sequence.append(potentials)
            
        return clock_sequence


class ROETrapPumping(ROE):
        
    def _generate_clock_sequence(self, n_steps, integration_step=0):

        """ 
        
        WIP
        
        
        A clocking sequence for trap pumping through n_steps/2 phases, with one phase
        always held high. The starting 
        """

        # The number of steps need not be the same as the number of phases 
        # in the CCD, but is for a simple scheme.
        assert ( n_steps % 2 ) == 0, "Expected n_steps to be even for trap pumping sequence"
        n_phases = n_steps / 2
        
        clock_sequence = []
        for step in range(n_steps):
            potentials = []
            for phase in range(n_phases):
                if step <= ( n_phases / 2 ): 
                    high_phase = ( integration_step + step ) % n_phases
                else:
                    high_phase = ( integration_step - step ) % n_phases
                print(step,high_phase)
                high = (phase == high_phase)
                adjacent_phases_high = [phase]
                capture_from_which_pixel = 0
                n_steps_around_into_same_pixel = n_phases // 2 # 1 for 3 or 4 phase devices
                if phase > ( step + n_steps_around_into_same_pixel):
                    release_to_which_pixel = -1
                elif phase < (step -n_steps_around_into_same_pixel):
                    release_to_which_pixel = 1 # Release into charge cloud following (for normal readout)
                else:
                    release_to_which_pixel = 0 # Release into same charge cloud
                potential={'high':high,
                           'adjacent_phases_high':adjacent_phases_high,
                           'capture_from_which_pixel':capture_from_which_pixel,
                           'release_to_which_pixel':release_to_which_pixel}
                potentials.append(potential)
                #potentials.append(ROEPotential(
                #    high=high,
                #    adjacent_phases_high=adjacent_phases_high,
                #    capture_from_which_pixel=capture_from_which_pixel,
                #    release_to_which_pixel=release_to_which_pixel,
                #))
            clock_sequence.append(potentials)
            
        return clock_sequence


        
class ROEPotential(object):
    # Deprecated
    def __init__(
        self,
        high=True,
        adjacent_phases_high=[0],
        capture_from_which_pixel=0,
        release_to_which_pixel=0,
    ):
        """
        The configuration of readout electronics for one CCD phase, during 
        one step of a clocking sequence.
        """
        self.high = high
        if not isinstance(adjacent_phases_high, list):
            adjacent_phases_high = [adjacent_phases_high]
        self.adjacent_phases_high = adjacent_phases_high
        self.capture_from_which_pixel = capture_from_which_pixel
        self.release_to_which_pixel = release_to_which_pixel
        
        
