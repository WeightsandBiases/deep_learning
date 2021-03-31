import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        
        #i_t: input gate
        self.U_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))

        # f_t: the forget gate
        self.U_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))
        
        # c_t: the cell gate
        self.U_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_size))
        
        # o_t: the output gate
        self.U_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))



        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_len, sequence_len, feature_len = x.size()
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_len, self.hidden_size).to(x.device),
                torch.zeros(batch_len, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(sequence_len):
            x_t = x[:, t, :]
            # input 
            i_t = torch.sigmoid(torch.mm(x_t, self.U_i) + torch.mm(h_t, self.V_i) + self.b_i)
            f_t = torch.sigmoid(torch.mm(x_t, self.U_f) + torch.mm(h_t, self.V_f) + self.b_f)
            g_t = torch.tanh(torch.mm(x_t, self.U_c) + torch.mm(h_t, self.V_c) + self.b_c)
            o_t = torch.sigmoid(torch.mm(x_t, self.U_o) + torch.mm(h_t, self.V_o) + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

