from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):

            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                print(idx)
                v_t = self.grad_tracker[idx]['dw']
                v_t = self.momentum * v_t - self.learning_rate * m.dw
                m.weight = m.weight + v_t
                self.grad_tracker[idx]['dw'] = v_t
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                b_t = self.grad_tracker[idx]['db']
                b_t = self.momentum * b_t - self.learning_rate * m.db
                m.bias = m.bias + b_t
                self.grad_tracker[idx]['db'] = b_t
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################