from autograd import Variable, Module, MatrixMultiplication, Sigmoid
import numpy as np


class Linear(Module):
    def __init__(self, dims=(), var_constructor=Variable, use_torch=False):
        super(Linear, self).__init__(var_constructor, use_torch)
        self.weight = var_constructor(np.random.rand(*dims))
        self.bias = var_constructor(np.random.rand(dims[0], 1))
        self.params = self.collect_parameters()
        self._matmul = MatrixMultiplication()

    def forward(self, _x):
        return self._matmul(self.weight, _x) + self.bias


class Perceptron(Module):
    def __init__(self, output_size, input_size):
        super(Perceptron, self).__init__()
        self.linear = Linear((output_size, input_size))
        self.linear_weight = self.linear.weight  # NOTE: this can be improved - collect weights automatically

        self.sigmoid = Sigmoid()
        self.params = self.collect_parameters()

    def forward(self, _x):
        return self.sigmoid(self.linear(_x))


if __name__ == "__main__":
    x = Variable(np.random.rand(4, 1))
    print(x.shape)
    p = Perceptron(x.shape[0], 2)
    a = p(x)
    print(a)
    a.backward()
    print(p)