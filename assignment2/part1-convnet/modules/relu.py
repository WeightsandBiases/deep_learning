import numpy as np

class ReLU:
    '''
    An implementation of rectified linear units(ReLU)
    '''
    def __init__(self):
        self.cache = None
        self.dx= None

    def forward(self, x):
        '''
        The forward pass of ReLU. Save necessary variables for backward
        :param x: input data
        :return: output of the ReLU function
        '''
        out = None
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        '''

        :param dout: the upstream gradients
        :return:
        '''
        x = self.cache
        self.dx = dout * (x >= 0)
