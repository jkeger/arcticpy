""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
import math
from copy import deepcopy


class ROE(object):
    def __init__(
        self,
        dwell_times=[1],
        charge_injection=False,
        empty_traps_at_start=True,
        empty_traps_between_columns=True,
        integration_step=0,  # Why should this matter?
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
        empty_traps_at_start : bool   (aka first_pixel_different)
            Only used outside charge injection mode. Allows for the first
            pixel-to-pixel transfer differently to the rest. Physically, this may be
            because the first pixel that a charge cloud finds itself in is
            guaranteed to start with empty traps; whereas every other pixel's traps
            may  have been filled by other charge.
            True:  begin the readout process with empty traps, so some electrons
                   in the input image are immediately lost. Because the first 
                   pixel-to-pixel transfer is inherently different from the rest,
                   that transfer for every pixel is modelled first. In most 
                   situations, this makes it a factor ~(E+3)/(E+1) slower to run..
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

    def _generate_clock_sequence(self, n_steps, integration_step=0):

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

    def express_matrix_from_rows_and_express(self,
            rows,
            express=0,
            offset=0,
            dtype=float,
            n_active_pixels=0,
            #charge_injection=False,
            #n_active_pixels=0,
            #first_pixel_different=True,
        ):
        """ 
        To reduce runtime, instead of calculating the effects of every 
        pixel-to-pixel transfer, it is possible to approximate readout by 
        processing each transfer once (Anderson et al. 2010) or a few times 
        (Massey et al. 2014, section 2.1.5), then multiplying the effect of
        that transfer by the number of transfers it represents. This function
        computes the multiplicative factor, and returns it in a matrix that can 
        be easily looped over.

        Parameters
        ----------
        rows : int or list or range
            int:         The number of rows in an image.
            list/range:  A subset of rows in the image to be processed, for speed.
        express : int
            Parameter determining balance between accuracy (high values or zero)  
            and speed (low values).
                express = 0 (default, alias for express = n_pixels)
                express = 1 (fastest) --> compute the effect of each transfer once
                express = 2 --> compute n times the effect of those transfers that 
                          happen many times. After a few transfers (and e.g. eroded 
                          leading edges), the incremental effect of subsequent 
                          transfers can change.
                express = n_pixels (slowest) --> compute every pixel-to-pixel transfer
            Runtime scales as O(express^2).
        
        Optional parameters
        -------------------
        dtype : "int" or "float"
            Old versions of this algorithm assumed (unnecessarily) that all express 
            multipliers must be integers. It is slightly more efficient if this
            requirement is dropped, but the option to force it is included for
            backwards compatability.
        offset : int >=0
            Consider all pixels to be offset by this number of pixels from the 
            readout register. Useful if working out the matrix for a postage stamp 
            image, or to account for prescan pixels whose data is not stored.
        n_active_pixels : int
            Used only in charge injection mode, this specifies the number of pixels
            from the readout register to the farthest pixel. It is useful as a
            keyword if the image supplied is not the entire area, either because it
            is a postage stamp (whose final pixel is not the end of the image), or
            excludes pre-/over-scan pixels.

        Returns
        -------
        express_matrix : [[float]]
            The express multiplier values for each pixel-to-pixel transfer.
        """


        # Parse inputs
        # Assumed number of rows in the (entire) image (even if given a postage stamp).
        #  with enough rows to contain the supposed image, including offset
        window = range(rows) if isinstance(rows, int) else rows
        n_rows = max(window) + 1 + offset
        # Default to very slow but accurate behaviour
        if express == 0:
            express = n_rows - offset
        if self.charge_injection:
            assert (
                self.empty_traps_at_start == True
            ), "Traps must be assumed initially empty in CIL mode (what else would they be filled with?)"
            n_rows = max(n_rows, n_active_pixels)
        elif self.empty_traps_at_start:
            n_rows -= 1

        # Initialise an array
        express_matrix = np.zeros((express, n_rows), dtype=dtype)

        # Compute the multiplier factors
        max_multiplier = n_rows / express
        # if it's going to be an integer, ceil() rather than int() is required in case the number of rows is not an integer multiple of express
        if dtype == int:
            max_multiplier = math.ceil(max_multiplier)
        if self.charge_injection:
            if dtype == int:
                for i in reversed(range(express)):
                    express_matrix[i, :] = util.set_min_max(
                        max_multiplier, 0, n_rows - sum(express_matrix[:, 0])
                    )
            else:
                express_matrix[:] = max_multiplier
            # if dtype == int:
            #    express_matrix[0,:] -= sum(express_matrix[:,0]) - n_rows
        else:
            # This populates every row in the matrix with a range from 1 to n_rows
            express_matrix[:] = np.arange(1, n_rows + 1)
            # Offset each row to account for the pixels that have already been read out
            for express_index in range(express):
                express_matrix[express_index] -= express_index * max_multiplier
            # Set all values to between 0 and max_multiplier
            express_matrix[express_matrix < 0] = 0
            express_matrix[express_matrix > max_multiplier] = max_multiplier

            # Separate out the first transfer of every pixel, so that it can be different from subsequent transfers (it sees only empty traps)
            if self.empty_traps_at_start:
                # What we calculated before contained (intentionally) one transfer too few. Store it...
                express_matrix_small = express_matrix
                # ..then create a new matrix with one extra transfer for each pixel
                n_rows += 1
                express_matrix = np.flipud(np.identity(n_rows))
                # Combine the two sets of transfers appropriately
                for i in range(express):
                    n_nonzero = sum(express_matrix_small[i, :] > 0)
                    express_matrix[n_nonzero, 1:] += express_matrix_small[i, :]

        # Extract the section of the array corresponding to the image (without the offset)
        n_rows -= offset
        express_matrix = express_matrix[:, offset:]  # remove the offset
        express_matrix = express_matrix[:, window]  # keep only the region of interest
        express_matrix = express_matrix[  # remove all rows containing only zeros, for speed later
            np.sum(express_matrix, axis=1) > 0, :
        ]

        # Adjust so that each computed transfer has equal effect (most useful if offset != 0). Could have got here directly.
        if dtype is not int:
            for i in range((express_matrix.shape)[1]):
                first_one = (np.where(express_matrix[:, i] == 1))[0]
                express_matrix[first_one, i] = 0
                nonzero = express_matrix[:, i] > 0
                if sum(nonzero) > 0:
                    express_matrix[nonzero, i] = sum(express_matrix[nonzero, i]) / sum(
                        nonzero
                    )
                express_matrix[first_one, i] = 1

        # Can't yet compute the when_to_store matrix, in case we are dealing with multi-phase readout
        return express_matrix

    def when_to_store_traps_from_express_matrix(self, express_matrix):
        """
        Accounting for first transfer differently
        Decide appropriate moments to store trap occupancy levels, so the next
        EXPRESS iteration can continue from an (approximately) suitable configuration
        """
        (n_express, n_rows) = express_matrix.shape
        when_to_store_traps = np.zeros((n_express, n_rows), dtype=bool)
        if not self.empty_traps_at_start and not self.charge_injection:
            for express_index in range(n_express - 1):
                for row_index in range(n_rows - 1):
                    if express_matrix[express_index + 1, row_index + 1] > 0:
                        break
                when_to_store_traps[express_index, row_index] = True
        return when_to_store_traps



class ROETrapPumping(ROE):
    def _generate_clock_sequence(self, n_steps, integration_step=0):

        """ 
        
        WIP
        
        
        A clocking sequence for trap pumping through n_steps/2 phases, with one phase
        always held high. The starting 
        """

        assert( n_steps == 6 ), "Have only coded for 3-phase so far"
        assert(( n_steps % 2 ) == 0 ), "Expected n_steps to be even for trap pumping sequence"
        n_phases = n_steps // 2 
        
        clock_sequence = []
        for step in range(n_steps):
            potentials = []
            for phase in range(n_phases):
                step_prime = integration_step + abs( ( ( step + n_phases ) % n_steps ) - n_phases )# 0,1,2,3,2,1
                high_phase = step_prime % n_phases
                capture_from_which_pixel = ( step_prime + 1 - phase ) // 3
                release_to_which_pixel = capture_from_which_pixel
                high = phase == high_phase
                adjacent_phases_high = [high_phase]
                potential = {
                    "high": high,
                    "adjacent_phases_high": adjacent_phases_high,
                    "capture_from_which_pixel": capture_from_which_pixel,
                    "release_to_which_pixel": release_to_which_pixel,
                }
                potentials.append(potential)
                #print(step,step_prime,phase,high_phase,capture_from_which_pixel)
            clock_sequence.append(potentials)

        return clock_sequence

    def when_to_store_traps_from_express_matrix(self, express_matrix):
        """
        Accounting for first transfer differently
        Decide appropriate moments to store trap occupancy levels, so the next
        EXPRESS iteration can continue from an (approximately) suitable configuration
        """
        when_to_store_traps = np.ones(express_matrix.shape, dtype=bool)
        return when_to_store_traps
