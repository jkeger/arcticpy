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
        charge_injection=False,
        empty_traps_at_start=True,
        empty_traps_between_columns=True,
        integration_step=None,  # Why should this matter?
    ):
        """
        The parameters for the clocking of electrons (in an n-type CCD; the algorithm 
        will work equally well for holes in a p-type CCD). with read-out electronics.

        Inputs:
        -------
        dwell_times : float or [float]
            The time between steps in the clocking sequence, IN THE SAME UNITS 
            AS THE TRAP CAPTURE/RELEASE TIMESCALES. This can be a single float
            for single-step clocking, or a list for multi-step clocking; the
            number of steps in the clocking sequence is taken from the length of
            this list. The default value, [1], produces instantaneous transfer 
            between adjacent pixels, with no intermediate phases.
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
            False: that happens in some pixels but not all (the fractions depend
                   upon EXPRESS.
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
        
        Outputs:
        --------
        dwell_times: [float]
            The time between steps in the clocking sequence.
        n_steps: int
            The number of steps in the clocking sequence.
        n_phases: int
            The assumed number of phases in the CCD. This is determined by the 
            clock sequence, which is defined internally by _generate_clock_sequence()
        clock_sequence: [[dict]]
            A description of the electrostatic potentials in each phase of the CCD, 
            during each step of a clock sequence. Unlike CCD pixels, where row 1 is 
            closer to the readout register than row 2, phase 2 is closer than phase 1.
            The dictionaries can be referenced as 
            roe.clock_sequence[clocking_step][phase] and contain: 
                high: bool
                    Is the potential held high, i.e. able to contain free electrons?
                adjacent_phases_high: [int]
                
                capture_from_which_pixel: int
                
                release_to_which_pixel: int         

        min/max_referred_to_pixel:
            The relative row number of the most distant charge cloud from which 
            electrons are captured or to which electrons are released, at any point 
            during the clock sequence.
                        
        """

        # Check input number of dwell times as an indication of the desired
        # number of steps in the clocking sequence
        if not isinstance(dwell_times, list):
            dwell_times = [dwell_times]
        self.n_steps = len(dwell_times)
        self.dwell_times = dwell_times

        self.charge_injection = charge_injection
        self.empty_traps_at_start = empty_traps_at_start
        self.empty_traps_between_columns = empty_traps_between_columns

        self.clock_sequence = self._generate_clock_sequence(
            self.n_steps, integration_step=integration_step
        )
        self.n_phases = len(self.clock_sequence[0])
        self.integration_step = integration_step

        # Record which range of charge clouds are accessed during the clocking sequence
        referred_to_pixels = [0]
        for step in range(self.n_steps):
            for phase in range(self.n_phases):
                referred_to_pixels.extend(
                    [
                        self.clock_sequence[step][phase]["capture_from_which_pixel"],
                        self.clock_sequence[step][phase]["release_to_which_pixel"],
                    ]
                )
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
                high_phase = step  # n_phases - phase
                high = phase == high_phase
                adjacent_phases_high = [high_phase]
                capture_from_which_pixel = 0
                n_steps_around_into_same_pixel = (
                    n_phases // 2
                )  # 1 for 3 or 4 phase devices
                if phase > (step + n_steps_around_into_same_pixel):
                    release_to_which_pixel = -1
                elif phase < (step - n_steps_around_into_same_pixel):
                    release_to_which_pixel = (
                        1  # Release into charge cloud following (for normal readout)
                    )
                else:
                    release_to_which_pixel = 0  # Release into same charge cloud
                potential = {
                    "high": high,
                    "adjacent_phases_high": adjacent_phases_high,
                    "capture_from_which_pixel": capture_from_which_pixel,
                    "release_to_which_pixel": release_to_which_pixel,
                }
                potentials.append(potential)
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
        assert (
            n_steps % 2
        ) == 0, "Expected n_steps to be even for trap pumping sequence"
        n_phases = n_steps / 2

        clock_sequence = []
        for step in range(n_steps):
            potentials = []
            for phase in range(n_phases):
                if step <= (n_phases / 2):
                    high_phase = (integration_step + step) % n_phases
                else:
                    high_phase = (integration_step - step) % n_phases
                print(step, high_phase)
                high = phase == high_phase
                adjacent_phases_high = [phase]
                capture_from_which_pixel = 0
                n_steps_around_into_same_pixel = (
                    n_phases // 2
                )  # 1 for 3 or 4 phase devices
                if phase > (step + n_steps_around_into_same_pixel):
                    release_to_which_pixel = -1
                elif phase < (step - n_steps_around_into_same_pixel):
                    release_to_which_pixel = (
                        1  # Release into charge cloud following (for normal readout)
                    )
                else:
                    release_to_which_pixel = 0  # Release into same charge cloud
                potential = {
                    "high": high,
                    "adjacent_phases_high": adjacent_phases_high,
                    "capture_from_which_pixel": capture_from_which_pixel,
                    "release_to_which_pixel": release_to_which_pixel,
                }
                potentials.append(potential)
                # potentials.append(ROEPotential(
                #    high=high,
                #    adjacent_phases_high=adjacent_phases_high,
                #    capture_from_which_pixel=capture_from_which_pixel,
                #    release_to_which_pixel=release_to_which_pixel,
                # ))
            clock_sequence.append(potentials)

        return clock_sequence
