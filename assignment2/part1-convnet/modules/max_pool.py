import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        kernel_h, kernel_w = (self.kernel_size, self.kernel_size)
        H_out = int(1 + (H - kernel_h) / self.stride)
        W_out = int(1 + (W - kernel_w) / self.stride)
        self.cache = (x, H_out, W_out)
        out = np.zeros((N, C, H_out, W_out))
        
        for img_i in range(N):
            for c_i in range(C):
                for j, r in enumerate(range(0, H, self.stride)):
                    for i, c in enumerate(range(0, W, self.stride)):
                        out[img_i, c_i, j, i] = np.max(x[img_i, c_i, r:r+kernel_h, c:c+kernel_w])
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        N, C, H, W = x.shape

        self.dx = np.zeros(x.shape)
        kernel_h, kernel_w = (self.kernel_size, self.kernel_size)

        for img_i in range(N):
            for c_i in range(C):
                for j, r in enumerate(range(0, H, self.stride)):
                    for i, c in enumerate(range(0, W, self.stride)):
                        pool = x[img_i, c_i, r: r+kernel_h, c: c+kernel_w]
                        mask = (pool == np.max(pool))
                        self.dx[img_i, c_i, r:r+kernel_h, c:c+kernel_w] = mask*dout[img_i, c_i, j, i]

