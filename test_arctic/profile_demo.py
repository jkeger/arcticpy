""" Profile the code to test and debug the performance. """

import os
import numpy as np
from astropy.io import fits

import arctic as ac


@ac.profile
def profile_HST_column():
    
    print("test")
    
    return 0
    
    
if __name__ == "__main__":
    profile_HST_column()