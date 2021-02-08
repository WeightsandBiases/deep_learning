class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        '''
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        '''

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        # credit to 
        # https://stats.stackexchange.com/questions/259752/sgd-l2-penalty-weights-update
        # directly apply loss regularization to gradient
        for w_k in model.weights.keys():
            # only apply L2 reg on weight gradients, not biases
            if 'W' in w_k:
                model.gradients[w_k] += self.reg * model.weights[w_k]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################