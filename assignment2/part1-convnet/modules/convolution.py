import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def convolve_2D(self, img, kernel, out, img_i, channel_out_i):
        if len(img.shape) == 2:
            img_h, img_w = img.shape
        elif len(img.shape) == 3:
            img_c, img_h, img_w = img.shape
        kernel_h, kernel_w = (self.kernel_size, self.kernel_size)
        for i, x in enumerate(range(0, img_w - kernel_w + 1, self.stride)):
            for j, y in enumerate(range(0, img_h - kernel_h + 1, self.stride)):
                for c in range(img_c):
                    img_slice = img[c, y: y + kernel_h, x: x + kernel_w]
                    out[img_i, channel_out_i, j, i] += np.dot(kernel[c].flatten(), img_slice.flatten())
        # add bias
        out[img_i, channel_out_i] += self.bias[channel_out_i]

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        
        n_pad = ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding))
        x_padded = np.pad(x, pad_width=n_pad, mode='constant')
        self.cache = x_padded
        N, C, H, W = x.shape
        kernel_h, kernel_w = (self.kernel_size, self.kernel_size)
        H_prime = int(H - kernel_h + 2 * self.padding / self.stride + 1)
        W_prime = int(W - kernel_w + 2 * self.padding / self.stride + 1)
        out = np.zeros((N, self.out_channels, H_prime, W_prime))
        for img_i in range(x.shape[0]):
            for channel_out_i in range(self.out_channels):
                img = x_padded[img_i]
                kernel = self.weight[channel_out_i]
                self.convolve_2D(img, kernel, out, img_i, channel_out_i)
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        input_padded = self.cache
        N, out_C, H_prime, W_prime = dout.shape
        img_c, img_h, img_w = input_padded[0].shape
        kernel_w, kernel_h = (self.kernel_size, self.kernel_size)
        self.db = np.sum(dout, axis=(0, 2, 3))
        self.dw = np.zeros((self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size))
        self.dx = np.zeros((N, self.in_channels, img_h, img_w))
        n_pad = ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding))
        d_out_padded = np.pad(dout, pad_width=n_pad, mode='constant')
        print(dout.shape)
        print("KERNEL SIZE", self.kernel_size)
        for img_i in range(N):
            for out_C_i in range(out_C):
                for i, x in enumerate(range(0, img_w - W_prime + 1, self.stride)):
                    for j, y in enumerate(range(0, img_h - H_prime + 1, self.stride)):
                        for in_C_i in range(img_c):
                            img_slice = input_padded[img_i, in_C_i, y: y + H_prime, x: x + W_prime]
                            self.dw[out_C_i, in_C_i, j, i] += np.dot(img_slice.flatten(), dout[img_i, out_C_i].flatten())
                for i, x in enumerate(range(0, img_w - kernel_w + 1, self.stride)):
                    for j, y in enumerate(range(0, img_h - kernel_h + 1, self.stride)):
                        for in_C_i in range(img_c):
                            kernel = self.weight[out_C_i, in_C_i]
                            kernel_180 = np.rot90(np.rot90(kernel))
                            dout_padded_slice = d_out_padded[img_i, out_C_i, y: y + kernel_h, x: x + kernel_w]
                            self.dx[img_i, in_C_i, j, i] += np.dot(dout_padded_slice.flatten(), kernel_180.flatten())
                

        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################