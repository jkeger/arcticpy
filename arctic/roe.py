""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy


class ROE(object):
    def __init__(
        self,
        sequence=[1],
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
            Not yet implemented
        """

        # Make sure the arrays are arrays
        if not isinstance(sequence, list):
            sequence = [sequence]

        self.sequence = sequence
        self.charge_injection = charge_injection
        self.empty_traps_at_start = empty_traps_at_start
        self.empty_traps_between_columns = empty_traps_between_columns
