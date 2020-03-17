""" CTI python

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    WIP...
"""
import numpy as np
from copy import deepcopy

from arctic.traps import (
    TrapNonUniformHeightDistribution,
    TrapManager,
    TrapManagerNonUniformHeightDistribution,
)


class Clocker(object):
    def __init__(
        self,
        sequence=[1],
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
        """

        # Make sure the arrays are arrays
        if not isinstance(sequence, list):
            parallel_sequence = [parallel_sequence]

        self.sequence = sequence
