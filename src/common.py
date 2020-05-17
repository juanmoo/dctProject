# Parent class to be used in the different compression methods

import numpy as np

class Encoding:

    def __init__(self, img):
        self.img = np.array(img)

    '''
    Attempts to create encoded version of 
    '''
    def encode(self, **args):
        raise NotImplementedError()

    '''
    Gives size of compressed object as number of floats/ints stored.
    '''
    def calculate_size(self):
        raise NotImplementedError()
