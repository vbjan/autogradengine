from . import Variable, Module, MatrixMultiplication, Tanh, Sigmoid
import numpy as np


class Linear(Module):
    def __init__(self, dims=(), var_constructor=Variable, use_torch=False):
        """
        Autograd Module to create a linear layer
        $$f(x) = Wx + b$$
        @param dims: (output_size, input_size)
        """
        super(Linear, self).__init__(var_constructor, use_torch)
        self.weight = var_constructor(np.random.rand(*dims))
        self.bias = var_constructor(np.random.rand(dims[0], 1))
        self._matmul = MatrixMultiplication()

    def forward(self, _x):
        return self._matmul(self.weight, _x) + self.bias


class Perceptron(Module):
    def __init__(self, output_size, input_size):
        """
        Autograd Module to create a perceptron
        """
        super(Perceptron, self).__init__()
        self.linear = Linear((output_size, input_size))
        # self.acti = Sigmoid()
        self.acti = Tanh()

    def forward(self, _x):
        return self.acti(self.linear(_x))


class Dataset:
    def __init__(self, x_data, y_data):
        """
        Autograd Dataset - allows to use the forward_dataset method of a Module, to pass all elements of the dataset
        through the module

        @param x_data: np.array of shape (n_samples, m, n) - with n = 1 for vector input
        @param y_data: np.array of shape (n_samples, m, n) - with n = 1 for vector input
        """
        self.x_data = x_data
        self.y_data = y_data

    def get_zip(self):
        return zip(self.x_data, self.y_data)

    def __len__(self):
        return len(self.x_data)

    def __repr__(self):
        return f"Dataset(\nx_data=\n{self.x_data}, \ny_data=\n{self.y_data})"


if __name__ == "__main__":
    x = Variable(np.random.rand(4, 1))
    print(x.shape)
    p = Perceptron(x.shape[0], 2)
    a = p(x)
    print(a)
    a.backward()
    print(p)