'''
Utility functions
'''

import numpy as np

def nextpow2(x):
    return 2**np.ceil(np.log2(x))
