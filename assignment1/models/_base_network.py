# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''

        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        # credit to Piazza forums and https://github.com/parasdahal/deepnet/blob/master/deepnet/utils.py?fbclid=IwAR3qLC1Njq0L_rc7JmCWVuhWuKerzjuofYmvLQVzn9QnUyZmhx9XBkGW9f4
        # for implementation of numerically stable softmax function
        exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        y_len = len(y)
        cross_entropy_i = -np.log(x_pred[range(y_len), y])
        # print(y_len)
        return np.sum(cross_entropy_i) / y_len

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        x_max = np.argmax(x_pred, axis=1)
        # print(len(y))
        return np.sum(x_max == y) / len(y) 

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        return 1.0 / (1.0 + np.exp(-X))

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################

        return self.sigmoid(x)*(1-self.sigmoid(x))

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        
        return np.maximum(0, X)

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return 1.0 * (X>0)
