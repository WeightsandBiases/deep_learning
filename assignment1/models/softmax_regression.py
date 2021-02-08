# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

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
        forward pass of softmax
        :param X: a batch of image (N, 28 x 28)
        '''
        self.layer_1 = np.matmul(X, self.weights['W1'])
        self.layer_1_ReLu = self.ReLU(self.layer_1)
        return self.softmax(self.layer_1_ReLu)

    def backward_pass(self, X, x_pred, y):
        '''
        backward pass of softmax
        :param x_pred: prediction result
        :param y: truth labels
        '''
        # dL/dW = dL/d_xentropy * d_xentropy/d_ReLU*dReLU/dWX*dX/dW
        # dL/d_xentropy
        grad_layer_1_out = self.delta_cross_entropy(self.layer_1_ReLu, y)
        # d_xentropy/d_ReLU
        grad_layer_1 = self.ReLU_dev(self.layer_1)
        dL_dWx = np.multiply(grad_layer_1_out, grad_layer_1)
        dL_dW = np.matmul(np.transpose(X), dL_dWx)

        self.gradients['W1'] = dL_dW


    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient with respect to the loss                       #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################

        x_pred = self.forward_pass(X)
        loss = self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)
        
        if mode != 'train':

            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################

        self.backward_pass(X, x_pred, y)

        return loss, accuracy





        


