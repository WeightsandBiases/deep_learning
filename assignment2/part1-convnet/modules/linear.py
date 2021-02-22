import numpy as np

class Linear:
    '''
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    '''
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None


    def forward(self, x):
        '''
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where sum of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        '''
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        x = x.reshape((x.shape[0], self.in_dim))
        self.cache = np.matmul(x, self.weight) + self.bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return self.cache

    def backward(self, dout):
        '''
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        self.dx = self.weight
        self.db = 0
        self.dw = x
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
