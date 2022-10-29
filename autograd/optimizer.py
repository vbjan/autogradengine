class GD:
    def __init__(self, lr=0.01, params=None):
        """
        Gradient Descent Optimizer for autograd.Modules

        @param lr: learning rate
        @param params: set of parameters of autograd.Modules that require gradient updates
        """
        self.lr = lr
        self.params = params

    def zero_grad(self):
        """
        Set all gradients to zero to makesure that the gradients are not accumulated with multiple backward passes
        """
        for param in self.params:
            param.grad = 0

    def step(self):
        """
        Update the parameters of the autograd.Module
        """
        for param in self.params:
            # all gradients are saved as Jacobian matrices which is why we need to transpose
            param.value -= self.lr * param.grad.transpose()
