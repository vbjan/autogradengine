from autograd import Variable, Module, MatrixMultiplication, Sigmoid
import numpy as np


class Linear(Module):
    def __init__(self, dims=(), var_constructor=Variable, use_torch=False):
        super(Linear, self).__init__(var_constructor, use_torch)
        self.weights = var_constructor(np.random.rand(*dims))
        self.params = self.collect_parameters()
        self._matmul = MatrixMultiplication()

    def forward(self, x):
        return self._matmul(self.weights, x)


class Perceptron(Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        self.linear = Linear((output_size, input_size))
        self.linear_weights = self.linear.weights  # NOTE: this can be improved - collect weights automatically

        self.sigmoid = Sigmoid()
        self.params = self.collect_parameters()

    def forward(self, x):
        return self.sigmoid(self.linear(x))





if __name__ == "__main__":
    x = Variable(np.random.rand(4, 1))
    print(x.shape)
    p = Perceptron(x.shape[0], 2)
    a = p(x)
    print(a)
    a.backward()
    print(p)