# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def delta_cross_entropy(self, x_pred, y):
        '''
        calculate the derivative of cross entropy combined with softmax
        credit to Piazza forums and 
        https://deepnotes.io/softmax-crossentropy?fbclid=IwAR1FQb_iaO0dqbjanigoUy0116VkOrxuWu_hd0--Y1ioi03Vlk0FgN0KFIM#derivative-of-cross-entropy-loss-with-softmax
        for derivation of cross entropy and softmax gradients
        :param x_pred: predictions of forward pass
        :param y: truth labels
        :return
            gradient: gradient of the cross entropy and softmax terms
        '''
        y_len = len(y)
        gradient = self.softmax(x_pred)
        gradient[range(y_len), y] -= 1
        gradient /= y_len
        return gradient

    def forward_pass(self, X):
        '''
        :param X: a batch of images (N, input size)
        :return:
            result of foward pass
        '''
        # muptiply layer by weights plus bias
        self.layer_1 = np.matmul(X, self.weights['W1']) + self.weights['b1']
        # apply activation function between layer 1 and 2
        self.layer_1_out = self.sigmoid(self.layer_1)
        self.layer_2 = np.matmul(self.layer_1_out, self.weights['W2']) + self.weights['b2']
        # apply softmax function before cross entropy
        return self.softmax(self.layer_2)

    def backward_pass(self, X, x_pred, y):
        '''
        :param output: the output of the forward pass 
        :return:
            none, but updates the internal weights
        '''
        # d_L/d_w = d_Z/d_W * d_L/d_Z
        # layer 2
        dL_dB = self.delta_cross_entropy(self.layer_2, y)
        self.gradients['W2'] = np.matmul(np.transpose(self.layer_1_out), dL_dB)
        self.gradients['b2'] = np.sum(dL_dB, axis=0)
        # layer 1
        # dL/dW1 = dZ_1/dW_1 * dA/dZ_1 * dB/dA * dL/dB
        dB_dA = self.weights['W2']
        dA_dZ_1 = self.sigmoid_dev(self.layer_1)
        dZ_1_dW_1 = np.transpose(X)
        # all together now
        dL_dA = np.matmul(dB_dA, dL_dB.T)
        dL_dZ_1 = np.multiply(dA_dZ_1, dL_dA.T)
        self.gradients['W1'] = np.matmul(dZ_1_dW_1, dL_dZ_1)
        self.gradients['b1'] = np.sum(dL_dZ_1, axis=0)
        

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        x_pred = self.forward_pass(X)
        loss = self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':

            return loss, accuracy

        self.backward_pass(X, x_pred, y)

        return loss, accuracy


