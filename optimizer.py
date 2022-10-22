
class GD:
    def __init__(self, lr=0.01, params=[]):
        self.lr = lr
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
