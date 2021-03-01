import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self._N = 128
        self._in_channels = 3
        self._H = 32
        self._W = 32
        # input group convolution
        self._ki = 3  # kernel size
        self._ki_p = 0  # padding
        self._ki_s = 1  # stride
        self._ki_oc = 16  # out channels
        # group 1 convolution
        self._k1 = 3  # kernel size
        self._k1_p = 0  # padding
        self._k1_s = 1  # stride
        self._k1_oc = 32  # out channels
        # group 2 maxpool
        self._p2 = 2  # pool2d kernel size
        self._p2_s = 2  # pool2d stride
        # group 2 convolution
        self._k2 = 3  # kernel size
        self._k2_p = 0  # padding
        self._k2_s = 1  # stride
        self._k2_oc = 64  # out channels
        # group 3-1 convolution
        self._k3_1 = 3  # kernel size
        self._k3_1_p = 0  # padding
        self._k3_1_s = 1  # stride
        self._k3_1_oc = 128  # out channels
        # group 3-2 convolution
        self._k3_2 = 3  # kernel size
        self._k3_2_p = 0  # padding
        self._k3_2_s = 1  # stride
        self._k3_2_oc = 256  # out channels
        # group 5-1 convolution
        self._k4_1 = 3  # kernel size
        self._k4_1_p = 0  # padding
        self._k4_1_s = 1  # stride
        self._k4_1_oc = 518  # out channels
        # group 5-2 convolution
        self._k4_2 = 3  # kernel size
        self._k4_2_p = 0  # padding
        self._k4_2_s = 1  # stride
        self._k4_2_oc = 1024  # out channels
        # fully connected layers
        self._n_classes = 10
        self._fc_1_out = 512
        self._fc_2_out = 128
        self._fc_3_out = 64
        in_size = (self._N, self._in_channels, self._H, self._W)
        device = self.get_device()
        # layer to normalize data
        self._normlayer = nn.LayerNorm(in_size[1:]).to(device)
        self._module_1 = nn.Sequential(
            nn.Conv2d(
                self._in_channels, self._ki_oc, self._ki, self._ki_s, self._ki_p,
            ),
            nn.BatchNorm2d(self._ki_oc),
            nn.ReLU(),
            nn.Conv2d(self._ki_oc, self._k1_oc, self._k1, self._k1_s, self._k1_p,),
            nn.BatchNorm2d(self._k1_oc),
            nn.ReLU(),
        ).to(device)
        # keep track of dims
        H_module_1, W_module_1 = self.get_out_dim_Conv2d(
            self._H, self._W, self._ki, self._ki, self._ki_p, self._ki_s,
        )
        H_module_1, W_module_1 = self.get_out_dim_Conv2d(
            H_module_1, W_module_1, self._k1, self._k1, self._k1_p, self._k1_s,
        )
        self._module_2 = nn.Sequential(
            nn.MaxPool2d(self._p2, self._p2_s),
            nn.Conv2d(self._k1_oc, self._k2_oc, self._k2, self._k2_s, self._k2_p,),
        )
        # keep track of dims
        H_module_2, W_module_2 = self.get_out_dim_MaxPool2d(
            H_module_1, W_module_1, self._p2, self._p2, self._p2_s,
        )
        H_module_2, W_module_2 = self.get_out_dim_Conv2d(
            H_module_2, W_module_2, self._k2, self._k2, self._k2_p, self._k2_s,
        )
        self._module_3 = nn.Sequential(
            nn.Conv2d(
                self._k2_oc, self._k3_1_oc, self._k3_1, self._k3_1_s, self._k3_1_p,
            ),
            nn.BatchNorm2d(self._k3_1_oc),
            nn.ReLU(),
            nn.Conv2d(
                self._k3_1_oc, self._k3_2_oc, self._k3_2, self._k3_2_s, self._k3_2_p,
            ),
            nn.BatchNorm2d(self._k3_2_oc),
            nn.ReLU(),
        ).to(device)
        # keep track of dims
        H_module_3, W_module_3 = self.get_out_dim_Conv2d(
            H_module_2, W_module_2, self._k3_1, self._k3_1, self._k3_1_p, self._k3_1_s,
        )
        H_module_3, W_module_3 = self.get_out_dim_Conv2d(
            H_module_3, W_module_3, self._k3_2, self._k3_2, self._k3_2_p, self._k3_2_s,
        )
        self._module_4 = nn.Sequential(
            nn.Conv2d(
                self._k3_2_oc, self._k4_1_oc, self._k4_1, self._k4_1_s, self._k4_1_p,
            ),
            nn.BatchNorm2d(self._k4_1_oc),
            nn.ReLU(),
            nn.Conv2d(
                self._k4_1_oc, self._k4_2_oc, self._k4_2, self._k4_2_s, self._k4_2_p,
            ),
            nn.BatchNorm2d(self._k4_2_oc),
            nn.ReLU(),
        ).to(device)
        # keep track of dims
        H_module_4, W_module_4 = self.get_out_dim_Conv2d(
            H_module_3, H_module_3, self._k4_1, self._k4_1, self._k4_1_p, self._k4_1_s,
        )
        H_module_4, W_module_4 = self.get_out_dim_Conv2d(
            H_module_4, W_module_4, self._k4_2, self._k4_2, self._k4_2_p, self._k4_2_s,
        )
        print("DIM4", H_module_4)
        self._dropout = nn.Dropout(inplace=True).to(device)
        self._fc_in_dim = self._k4_2_oc * H_module_4 * W_module_4
        self._fc = nn.Sequential(
            nn.Linear(self._fc_in_dim, self._fc_1_out),
            nn.Linear(self._fc_1_out, self._fc_2_out),
            nn.Linear(self._fc_2_out, self._fc_3_out),
            nn.Linear(self._fc_3_out, self._n_classes),
        ).to(device)


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
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        x = self._normlayer(x)
        x = self._module_1(x)
        x = self._module_2(x)
        x = self._module_3(x)
        x = self._module_4(x)
        x = self._dropout(x)
        x = x.view(-1, self._fc_in_dim)
        out = self._fc(x)
        return out
