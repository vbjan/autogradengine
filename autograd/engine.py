import numpy as np
import torch


class Variable:
    """
    Analog to torch.Tensor -> this is where the magic happens. All operations performed between Variables are represented
    as a comutation graph that is constructed dynamically. The graph is traversed backwards to calculate the gradients
    of all Variables with requires_grad=True. Each Variable is a Node in the graph and contains the references of its
    children, to allow for easy traversal during backpropagation.
    """
    def __init__(self, value, _children=(), _op=None, requires_grad=True):
        """
        @param value: value of Variable as np.array of shape (m, n)
        @param _children: children of Variable in computational graph
        @param _op: reference to Operation that created Variable in computational graph
        @param requires_grad: bool indicating whether gradients should be calculated for Variable
        """
        if is_scalar(value):
            self.value = np.array([[value]], dtype=float)
        else:
            self.value = np.array(value, dtype=float)
        self.shape = self.value.shape
        # all gradients are saved as Jacobian matrices which is why we need to transpose
        self.grad = np.zeros(self.shape).transpose()
        self._children = _children
        self._op = _op
        self.requires_grad = requires_grad

    def _local_backward(self):
        """
        Calculates the local gradient of self with respect to its children
        """
        assert self._op is not None
        if self._op.single_variable_op is True:
            gradient = self._op.df(self._children[0])
            self._children[0].grad += np.matmul(self.grad, gradient[0])
        else:
            gradient = self._op.df(self._children[0], self._children[1])
            if self._op.name == 'matmul':  # formulas are in the form of multiplying jacobians
                self._children[0].grad += np.matmul(self.grad.transpose(), gradient[0]).transpose()
                self._children[1].grad += np.matmul(self.grad, gradient[1])
            else:
                self._children[0].grad += np.matmul(self.grad, gradient[0])
                self._children[1].grad += np.matmul(self.grad, gradient[1])

    def backward(self):
        """
        Backpropagates the gradient of self with respect to all Variables with requires_grad=True through the
        computational graph
        """
        top_var_list = []
        visited = set()

        # do topological sort to allow iteration through variables to calculated local gradients of each variable
        def topological_sort(var):
            if var not in visited:
                visited.add(var)
                for child in var._children:
                    topological_sort(child)
                top_var_list.append(var)

        topological_sort(self)

        self.grad = np.identity(self.shape[0])
        for var in reversed(top_var_list):
            if var.requires_grad and var._op is not None:
                var._local_backward()

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, _op={self._op})"

    # OPERATOR OVERLOADING FOR Variable class
    # element-wise addition
    def __add__(self, other):
        return Addition.f(self, other)

    # used to allow operations like: int(1) + Variable(2)
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

    # useful methods to make manipulation of Variables easier
    def sign(self):
        if self.value.shape != 1:
            raise ValueError("Sign operation only works on scalars")
        if self.value < 0:
            return -1
        if self.value > 0:
            return 1
        else:
            return 0

    def sum(self):
        return Sum.f(self)

    def transpose(self):
        self.value = self.value.transpose()
        self.grad = self.grad.transpose()
        return self


class Module:
    """
    Analog to torch.nn.Module
    """
    def __init__(self, var_constructor=Variable, use_torch=False):
        """
        @param var_constructor: This allows to pass torch.Tensor (for gradient checking, see test.py)
        @param use_torch: Set to true when using PyTorch to calculate gradients (for gradient checking)
        """
        self.var_constructor = var_constructor
        self.use_torch = use_torch

    def __repr__(self):
        params = self.collect_parameters()
        description = 'Module(\n'
        for i, param in enumerate(params):
            description += f'grad_{i}={param.grad}, '
        return description + '\n)'

    def __eq__(self, other):  # To compare autograd Modules with torch.nn.Modules
        assert isinstance(other, Module)
        params = self.collect_parameters()
        for param, torch_param in zip(params, other.collect_parameters()):
            # Convert torch grad to np array to make it comparable
            _torch_grad = np.array(torch_param.grad)
            if not np.allclose(param.grad, _torch_grad, atol=1e-04):
                return False
        return True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_dataset(self, dataset, loss_function):
        loss = 0
        for x, y in dataset.get_zip():
            loss += loss_function(self(x), y)
        return loss/len(dataset)

    def collect_parameters(self):
        """
        Collects all parameters of the Module and any submodules used during initialization
        @return: set of all parameters that require gradients
        """
        if self.use_torch:
            # This only works for Modules that do not have any torch.nn.Modules
            params = [param for param in self.__dict__.values() if isinstance(param, torch.Tensor)]
        else:
            # Automatically collect all variables that are attributes of the module
            module_params = set(*tuple(param.collect_parameters() for param in self.__dict__.values() if isinstance(param, Module)))
            self_params = set(tuple(param for param in self.__dict__.values() if isinstance(param, Variable) and param.requires_grad))
            params = self_params | module_params  # union of sets
        return params


def is_scalar(var):
    if isinstance(var, int) or isinstance(var, float):
        return True


# DEFINE OPERATIONS - to allow to backpropagate through the computational graph
class Operation:
    """
    Base class for all operations.
    All operations support operations on 2d np.arrays )
    """
    def __init__(self):
        self.name = None
        self.single_variable_op = False

    @staticmethod
    def f(*args):
        """
        Calculate the output of the operation
        """
        raise NotImplementedError

    @staticmethod
    def df(*args):
        """
        Calculate the Jacobian of the operation
        """
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
    def make_into_vars(x, y):
        if is_scalar(y):  # transform scalar to Variable with same shape to allow broadcasting
            y = Variable(np.ones(x.value.shape) * y)
            y.grad = np.zeros(y.value.shape).transpose()
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        return x, y


# Two variable operations
class Addition(Operation):
    def __init__(self):
        super(Addition, self).__init__()
        self.name = '+'

    @staticmethod
    def f(x, y):  # cast y into tensor if needed
        x, y = Operation.make_into_vars(x, y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(x.value + y.value, _children=(x, y), _op=Addition())

    @staticmethod
    def df(x, y):
        shape = x.value.shape[0]
        full_grad = [np.zeros((shape, shape)), np.zeros((shape, shape))]
        if x.requires_grad:
            full_grad[0] = np.identity(shape)
        if y.requires_grad:
            full_grad[1] = np.identity(shape)
        return full_grad


class Subtraction(Operation):
    def __init__(self):
        super(Subtraction, self).__init__()
        self.name = '-'

    @staticmethod
    def f(x, y):
        x, y = Operation.make_into_vars(x, y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(x.value - y.value, _children=(x, y), _op=Subtraction())

    @staticmethod
    def df(x, y):
        shape = x.value.shape[0]
        full_grad = [np.zeros((shape, shape)), np.zeros((shape, shape))]
        if x.requires_grad:
            full_grad[0] = np.identity(shape)
        if y.requires_grad:
            full_grad[1] = -np.identity(shape)
        return full_grad


class Multiplication(Operation):
    def __init__(self):
        super(Multiplication, self).__init__()
        self.name = '*'

    @staticmethod
    def f(x, y):
        x, y = Operation.make_into_vars(x, y)
        if x.value.shape != y.value.shape:
            raise ValueError("Shape mismatch")
        return Variable(np.multiply(x.value, y.value), _children=(x, y), _op=Multiplication())

    @staticmethod
    def df(x, y):  # NOTE: derivatives work only for scalars and vectors not matrices!!!
        shape = x.value.shape[0]
        full_grad = [np.zeros((shape, shape)), np.zeros((shape, shape))]
        if x.requires_grad:
            full_grad[0] = np.diag(y.value.reshape(-1))
        if y.requires_grad:
            full_grad[1] = np.diag(x.value.reshape(-1))
        return full_grad


class Power(Operation):  # only works if power is scalar
    def __init__(self):
        super(Power, self).__init__()
        self.name = '**'

    @staticmethod
    def f(x, y):
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        assert(max(y.shape) == 1)  # only works if exponent is scalar!
        return Variable(np.power(x.value, y.value), _children=(x, y), _op=Power())

    @staticmethod
    def df(x, y):
        shape = x.value.shape[0]
        full_grad = [np.zeros((shape, shape)), np.array([0.])]
        assert(max(y.shape) == 1)  # only works if exponent is scalar!
        if x.requires_grad:
            full_grad[0] = np.multiply(y.value, np.power(x.value, (y.value - 1.)))
        if y.requires_grad:
            full_grad[1] = np.multiply(np.power(x.value, y.value), np.log(x.value))
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
        shape = x.value.shape[0]
        if x.requires_grad:
            return [-np.identity(shape)]
        return [np.zeros((shape, shape))]


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
            return [np.diag(np.divide(np.exp(x.value), np.power((1 + np.exp(x.value)), 2)).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


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
            return [np.diag(np.exp(x.value).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


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
            return [np.diag(np.divide(1, x.value).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


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
            return [np.diag(np.subtract(1, np.power(np.tanh(x.value), 2)).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


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
            return [np.diag(np.cos(x.value).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


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
            return [-np.diag(np.sin(x.value.squeeze()))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


class ReLu(Operation):
    def __init__(self):
        super(ReLu, self).__init__()
        self.name = 'relu'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.maximum(x.value, 0), _children=(x,), _op=ReLu())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.diag(np.greater(x.value, 0).astype(int).reshape(-1))]
        else:
            shape = x.value.shape[0]
            return [np.zeros((shape, shape))]


# MATRIX OPERATIONS:
class MatrixMultiplication(Operation):
    def __init__(self):
        super(MatrixMultiplication, self).__init__()
        self.name = 'matmul'
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
        full_jacobian = [np.zeros(y.value.transpose().shape), np.zeros(x.value.shape)]
        if x.requires_grad:
            full_jacobian[0] = y.value.transpose()
        if y.requires_grad:
            full_jacobian[1] = x.value
        return full_jacobian


class Sum(Operation):
    def __init__(self):
        super(Sum, self).__init__()
        self.name = 'sum'
        self.single_variable_op = True

    @staticmethod
    def f(x):
        x = Operation.check_if_var_else_create(x)
        return Variable(np.sum(x.value), _children=(x,), _op=Sum())

    @staticmethod
    def df(x):
        if x.requires_grad:
            return [np.ones(x.value.shape).transpose()]
        else:
            return [np.array([0.])]


# LOSS FUNCTIONS:
class MSELoss(Operation):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.name = 'mse'
        self.single_variable_op = False

    @staticmethod
    def f(x, y):
        x = Operation.check_if_var_else_create(x)
        y = Operation.check_if_var_else_create(y)
        return Variable(np.sum(np.power(x.value - y.value, 2)), _children=(x, y), _op=MSELoss())

    @staticmethod
    def df(x, y):
        full_jacobian = [np.zeros(x.value.transpose().shape), np.zeros(x.value.transpose().shape)]
        if x.requires_grad:
            full_jacobian[0] = 2 * (x.value - y.value).transpose()
        if y.requires_grad:
            full_jacobian[1] = -2 * (x.value - y.value).transpose()
        return full_jacobian