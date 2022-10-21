import numpy as np
import math


class Variable:
    def __init__(self, value, _children=(), _op=None, requires_grad=True):
        self.value = value
        self.grad = 0
        self._children = _children
        self._op = _op
        self.requires_grad = requires_grad
        
    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, _op={self._op})"

    def __add__(self, other):
        other = self.check_if_var_else_create(other)
        func = Addition()
        return func.f(self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self.check_if_var_else_create(other)
        func = Multiplication()
        return func.f(self, other)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = self.check_if_var_else_create(other)
        func = Subtraction()
        return func.f(self, other)

    def __rsub__(self, other):
        return self - other

    def __neg__(self):
        return Variable(-self.value, requires_grad=self.requires_grad)

    def __pow__(self, power):
        power = self.check_if_var_else_create(power)
        func = Power()
        return func.f(self, power)

    def sign(self):
        if self.value < 0:
            return -1
        if self.value > 0:
            return 1
        else:
            return 0

    @staticmethod
    def check_if_var_else_create(other):
        if not isinstance(other, Variable):
            # we don't calculate gradients for constants
            other = Variable(other, requires_grad=False)
        return other

    def _local_backward(self):
        assert self._op is not None
        if self._op.single_variable_op == True:
            gradient = self._op.df(self._children[0])
            self._children[0].grad += gradient[0] * self.grad
        else:
            gradient = self._op.df(self._children[0], self._children[1])
            self._children[0].grad += gradient[0] * self.grad
            self._children[1].grad += gradient[1] * self.grad

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

        self.grad = 1
        for var in reversed(top_var_list):
            if var.requires_grad and var._op is not None:
                var._local_backward()


class Operation:
    def __init__(self):
        self.name = None
        self.single_variable_op = False

    # *args must be of type Variable
    def f(self, *args):
        for v in args:
            assert isinstance(v, Variable)

    # *args must be of type Variable
    def df(self, *args):
        for v in args:
            assert isinstance(v, Variable)

    def __repr__(self):
        return f"Operation(name={self.name})"

    def __call__(self, *args):
        return self.f(*args)


class Addition(Operation):
    def __init__(self):
        super(Addition, self).__init__()
        self.name = '+'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value + y.value, _children=(x, y), _op=Addition())

    def df(self, x, y):
        super().df(x, y)
        return [1, 1]


class Subtraction(Operation):
    def __init__(self):
        super(Subtraction, self).__init__()
        self.name = '-'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value - y.value, _children=(x, y), _op=Subtraction())

    def df(self, x, y):
        super().df(x, y)
        return [1, -1]  


class Multiplication(Operation):
    def __init__(self):
        super(Multiplication, self).__init__()
        self.name = '*'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value * y.value, _children=(x, y), _op=Multiplication())

    def df(self, x, y):
        super().df(x, y)
        return [y.value, x.value]


class Power(Operation):
    def __init__(self):
        super(Power, self).__init__()
        self.name = '**'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value ** y.value, _children=(x, y), _op=Power())

    def df(self, x, y):
        super().df(x, y)
        return [y.value * x.value ** (y.value - 1), x.value ** y.value * np.log(x.value)]


class Sigmoid(Operation):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'sigmoid'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(1 / (1 + np.exp(-x.value)), _children=(x,), _op=Sigmoid())

    def df(self, x):
        super().df(x)
        return [np.exp(x.value) / (1 + np.exp(x.value)) ** 2]


class Module:
    def __init__(self, var, use_torch=False):
        self.var = None
        self.use_torch = use_torch
        self.params = []

    def __repr__(self):
        description = 'Module( '
        for i, param in enumerate(self.params):
            description += f'grad_{i}={float(param.grad):.2f}, '
        return description + ' )'

    def __eq__(self, other):
        assert isinstance(other, Module)
        for param, torch_param in zip(self.params, other.params):
            if not math.isclose(float(param.grad), float(torch_param.grad), abs_tol=1e-5):
                print(float(param.grad), float(torch_param.grad))
        return True



