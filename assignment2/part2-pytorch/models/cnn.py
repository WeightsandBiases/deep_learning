import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        # from the assignment docs
        self._in_channels = 3
        self._out_channels = 32
        self._kernel_size = 7
        self._stride = 1
        self._padding = 0
        self._pool_k_size = 2
        self._pool_stride = 2
        self._H = 32  # image height
        self._W = 32  # image width
        self._n_classes = 10  # classification classes
        device = self.get_device()
        self._layers = nn.Sequential(
            nn.Conv2d(
                self._in_channels,
                self._out_channels,
                self._kernel_size,
                self._stride,
                self._padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(self._pool_k_size, self._pool_stride),
        ).to(device)
        # dimensions of maxpool input after Conv2d
        H_MaxPool2d, W_MaxPool2d = self.get_out_dim_Conv2d(
            self._H,
            self._W,
            self._kernel_size,
            self._kernel_size,
            self._padding,
            self._stride,
        )
        # dimensions of fully connected layer after MaxPool2d
        H_fc, W_fc = self.get_out_dim_MaxPool2d(
            H_MaxPool2d,
            W_MaxPool2d,
            self._pool_k_size,
            self._pool_k_size,
            self._pool_stride,
        )
        self._fc_in_dim = self._out_channels * H_fc * W_fc
        self._fc_layer = nn.Linear(self._fc_in_dim, self._n_classes).to(device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def get_out_dim_Conv2d(self, H, W, k_h, k_w, padding, stride):
        H_out = int(H - k_h + 2 * padding / stride + 1)
        W_out = int(W - k_w + 2 * padding / stride + 1)
        return (H_out, W_out)

    def get_out_dim_MaxPool2d(self, H, W, k_h, k_w, stride):
        H_out = int(1 + (H - k_h) / stride)
        W_out = int(1 + (W - k_w) / stride)
        return (H_out, W_out)

    def get_device(self):
        """
        returns the device availible
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def forward(self, x):
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self._layers(x)
        x = x.view(-1, self._fc_in_dim)
        out = self._fc_layer(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return out
