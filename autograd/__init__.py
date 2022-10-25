import numpy as np
import math
import torch


class Variable:
    def __init__(self, value, _children=(), _op=None, requires_grad=True):
        if Operation.is_scalar(value):
            self.value = np.array([value], dtype=float)
        else:
            self.value = np.array(value, dtype=float)  # should be of type np.array
        self.shape = self.value.shape
        self.grad = np.array([0], dtype=float)
        self._children = _children
        self._op = _op
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, _op={self._op})"

    # element-wise addition
    def __add__(self, other):
        return Addition.f(self, other)

    def __radd__(self, other):
        return self + other

    # element-wise multiplication
    def __mul__(self, other):
        return Multiplication.f(self, other)

    def __rmul__(self, other):  # other * self
        return self * other

    # element-wise subtraction
    def __sub__(self, other):  # self - other
        return Subtraction.f(self, other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __neg__(self):
        return Negation.f(self)

    # element-wise power operation
    def __pow__(self, power):
        return Power.f(self, power)

    # element-wise division
    def __truediv__(self, other):
        return self * other ** -1.

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1.

    def sign(self):
        if self.value.shape != 1:
            raise ValueError("Sign operation only works on scalars")
        if self.value < 0:
            return -1
        if self.value > 0:
            return 1
        else:
            return 0

    def _local_backward(self):
        assert self._op is not None
        if self._op.single_variable_op == True:
            gradient = self._op.df(self._children[0])
            self._children[0].grad += gradient[0] * self.grad
        else:
            gradient = self._op.df(self._children[0], self._children[1])
            self._children[0].grad += np.matmul(self.grad, gradient[0])
            self._children[1].grad += np.matmul(gradient[1], self.grad)

    def backward(self):
        top_var_list = []
        visited = set()

        # do topological sort to allow iteration through variables to calculated local gradients of
        def topological_sort(var):
            if var not in visited:
                visited.add(var)
                for child in var._children:
                    topological_sort(child)
                top_var_list.append(var)

        topological_sort(self)

        self.grad = np.array([1], dtype=float)
        for var in reversed(top_var_list):
            if var.requires_grad and var._op is not None:
                var._local_backward()


class Module:
    def __init__(self, var_constructor=Variable, use_torch=False):
        self.var_constructor = var_constructor
        self.use_torch = use_torch
        self.params = None

    def __repr__(self):
        description = 'Module(\n'
        for i, param in enumerate(self.params):
            description += f'grad_{i}={param.grad}, '
        return description + '\n)'

    def __eq__(self, other):
        assert isinstance(other, Module)
        for param, torch_param in zip(self.params, other.params):
            if not math.isclose(float(param.grad), float(torch_param.grad), abs_tol=1e-5):
                return False
        return True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def collect_parameters(self):
        if self.use_torch:
            self.params = [param for param in self.__dict__.values() if isinstance(param, torch.Tensor)]
        else:
            self.params = [param for param in self.__dict__.values() if isinstance(param, Variable)]
        return self.params


# Just defining a bunch of operations:

# Elementwise operation taking two inputs
class Operation:
    def __init__(self):
        self.name = None
        self.single_variable_op = False

    @staticmethod
    def f(*args):
        raise NotImplementedError

    @staticmethod
    def df(*args):
        raise NotImplementedError

    def __repr__(self):
        return f"Operation(name={self.name})"

    # Calling operation also supports inputs not of type Variable
    def __call__(self, *args):
        return self.f(*args)

    @staticmethod
    def check_if_var_else_create(other):
        if not isinstance(other, Variable):
            # we don't calculate gradients for constants
            other = Variable(other, requires_grad=False)
        return other

    @staticmethod
    def is_scalar(var):
        if isinstance(var, int) or isinstance(var, float):
            return True


# Two variable operations
class Addition(Operation):
    def __init__(self):
        super(Addition, self).__init__()
        self.name = '+'

    @staticmethod
    def f(x, y):  # cast y into tensor if needed
        if Operation.is_scalar(y):
            y = Variable(np.ones(x.value.shape) * y)
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(x.value + y.value, _children=(x, y), _op=Addition())

    @staticmethod
    def df(x, y):
        full_grad = [np.array([0.]), np.array([0.])]
        if x.requires_grad:
            full_grad[0] = np.ones(x.value.shape)
        if y.requires_grad:
            full_grad[1] = np.ones(y.value.shape)
        return full_grad


class Subtraction(Operation):
    def __init__(self):
        super(Subtraction, self).__init__()
        self.name = '-'

    @staticmethod
    def f(x, y):
        if Operation.is_scalar(y):
            y = Variable(np.ones(x.value.shape) * y)
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(x.value - y.value, _children=(x, y), _op=Subtraction())

    @staticmethod
    def df(x, y):
        full_grad = [np.array([0.]), np.array([0.])]
        if x.requires_grad:
            full_grad[0] = np.ones(x.value.shape)
        if y.requires_grad:
            full_grad[1] = -np.ones(y.value.shape)
        return full_grad


class Multiplication(Operation):
    def __init__(self):
        super(Multiplication, self).__init__()
        self.name = '*'

    @staticmethod
    def f(x, y):
        if Operation.is_scalar(y):
            y = Variable(np.ones(x.value.shape) * y)
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(np.multiply(x.value, y.value), _children=(x, y), _op=Multiplication())

    @staticmethod
    def df(x, y):
        full_grad = [np.array([0.]), np.array([0.])]
        if x.requires_grad:
            full_grad[0] = y.value
        if y.requires_grad:
            full_grad[1] = x.value
        return full_grad


class Power(Operation):  # only works if power is scalar
    def __init__(self):
        super(Power, self).__init__()
        self.name = '**'

    @staticmethod
    def f(x, y):
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        return Variable(np.power(x.value, y.value), _children=(x, y), _op=Power())

    @staticmethod
    def df(x, y):
        full_grad = [np.array([0.]), np.array([0.])]
        if x.requires_grad:
            full_grad[0] = np.multiply(y.value, np.power(x.value, (y.value - 1.)))
        if y.requires_grad:
            full_grad[1] = np.multiply(np.power(x.value, y.value), np.log(x.value))
        #assert(isinstance(grad, np.ndarray) for grad in full_grad)
        return full_grad


# Single variable element wise operations:
class Negation(Operation):
    def __init__(self):
        super(Negation, self).__init__()
        self.name = '-'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(-x.value, _children=(x,), _op=Negation())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [-np.ones(x.value.shape)]
        return [np.array([0.])]


class Sigmoid(Operation):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'sigmoid'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.divide(1, (1 + np.exp(-x.value))), _children=(x,), _op=Sigmoid())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.divide(np.exp(x.value), np.power((1 + np.exp(x.value)), 2))]
        return [np.array([0.])]


class Exp(Operation):
    def __init__(self):
        super(Exp, self).__init__()
        self.name = 'exp'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.exp(x.value), _children=(x,), _op=Exp())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.exp(x.value)]
        return [np.array([0.])]


class Log(Operation):
    def __init__(self):
        super(Log, self).__init__()
        self.name = 'log'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.log(x.value), _children=(x,), _op=Log())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.divide(1, x.value)]
        return [np.array([0.])]


class Tanh(Operation):
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = 'tanh'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.tanh(x.value), _children=(x,), _op=Tanh())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.subtract(1, np.power(np.tanh(x.value), 2))]
        return [np.array([0.])]


class Sin(Operation):
    def __init__(self):
        super(Sin, self).__init__()
        self.name = 'sin'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.sin(x.value), _children=(x,), _op=Sin())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.cos(x.value)]
        return [np.array([0.])]


class Cos(Operation):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.cos(x.value), _children=(x,), _op=Cos())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [-np.sin(x.value)]
        return [np.array([0.])]


# Matrix operations:
class MatrixMultiplication(Operation):
    def __init__(self):
        super(MatrixMultiplication, self).__init__()
        self.name = '@'
        self.single_variable_op = False

    @staticmethod
    def f(x, y):
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        if x.value.shape[1] != y.value.shape[0]:
            raise ValueError(f"Shape mismatch x.shape={x.value.shape} and y.shape={y.value.shape}")
        return Variable(np.matmul(x.value, y.value), _children=(x, y), _op=MatrixMultiplication())

    @staticmethod
    def df(x, y):
        full_grad = [0, 0]
        if x.requires_grad:
            full_grad[0] = y.value.transpose()
        if y.requires_grad:
            full_grad[1] = x.value.transpose()
        return full_grad


if __name__ == '__main__':
    import time
    # Defining test function
    def gradient_test_f(x):
        return ((x - 3.) ** 2. / x - 1) ** (x - 1.)

    # Compute gradient df/dx at x=1 using torch
    torch_x = torch.Tensor([1.])
    torch_x.requires_grad = True

    torch_time = time.time()
    torch_y = gradient_test_f(torch_x)
    torch_y.backward()
    torch_time = time.time() - torch_time

    torch_grad = torch_x.grad

    # Compute gradient df/dx at x=1 using autograd
    autograd_x = Variable([1.])

    autograd_time = time.time()
    autograd_y = gradient_test_f(autograd_x)
    autograd_y.backward()
    autograd_time = time.time() - autograd_time

    autograd_grad = autograd_x.grad

    print(
        f"torch gradient {torch_grad.item():.3f} in {torch_time:.5f}s\nautograd gradient: {autograd_grad.item():.3f} in {autograd_time:.5f}s")
    if math.isclose(torch_grad, autograd_grad, abs_tol=1e-5):
        print("\nThe packages agree!")