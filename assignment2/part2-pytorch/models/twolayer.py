import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self._input_dim = input_dim
        device = self.get_device()
        # two layer network with sigmoid activation
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
            ).to(device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    def get_device(self):
        '''
        returns the device availible
        '''
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device

    def forward(self, x):
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        device = self.get_device()
        N = len(x)
        x = x.reshape((N, self._input_dim))
        out = self._model(x)
        # convert from one hot encoded to max of prediction
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out