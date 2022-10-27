class GD:
    def __init__(self, lr=0.01, params=None):
        self.lr = lr
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    def step(self):
        for param in self.params:
            # all gradients are saved as Jacobian matrices which is why we need to transpose
            param.value -= self.lr * param.grad.transpose()
