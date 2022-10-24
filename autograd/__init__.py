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

    def __rmul__(self, other):  # other * self
        return self * other

    def __sub__(self, other):  # self - other
        other = self.check_if_var_else_create(other)
        func = Subtraction()
        return func.f(self, other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):  # other / self
        return other * self ** (-1)

    def __neg__(self):
        func = Negation()
        return func.f(self)

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


class Module:
    def __init__(self, var=None, use_torch=False):
        self.var = var
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
                return False
        return True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Just defining a bunch of operations:
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

    # Calling operation also supports inputs not of type Variable
    def __call__(self, *args):
        new_args = ()
        for arg in args:
            new_args = new_args + (Variable.check_if_var_else_create(arg), )
        return self.f(*new_args)


# Two variable operations
class Addition(Operation):
    def __init__(self):
        super(Addition, self).__init__()
        self.name = '+'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value + y.value, _children=(x, y), _op=Addition())

    def df(self, x, y):
        super().df(x, y)
        full_grad = [0, 0]
        if x.requires_grad:
            full_grad[0] = 1
        if y.requires_grad:
            full_grad[1] = 1
        return full_grad


class Subtraction(Operation):
    def __init__(self):
        super(Subtraction, self).__init__()
        self.name = '-'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value - y.value, _children=(x, y), _op=Subtraction())

    def df(self, x, y):
        super().df(x, y)
        full_grad = [0, 0]
        if x.requires_grad:
            full_grad[0] = 1
        if y.requires_grad:
            full_grad[1] = -1
        return full_grad


class Multiplication(Operation):
    def __init__(self):
        super(Multiplication, self).__init__()
        self.name = '*'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value * y.value, _children=(x, y), _op=Multiplication())

    def df(self, x, y):
        super().df(x, y)
        full_grad = [0, 0]
        if x.requires_grad:
            full_grad[0] = y.value
        if y.requires_grad:
            full_grad[1] = x.value
        return full_grad


class Power(Operation):
    def __init__(self):
        super(Power, self).__init__()
        self.name = '**'

    def f(self, x, y):
        super().f(x, y)
        return Variable(x.value ** y.value, _children=(x, y), _op=Power())

    def df(self, x, y):
        super().df(x, y)
        full_grad = [0, 0]
        if x.requires_grad:
            full_grad[0] = y.value * x.value ** (y.value - 1)
        if y.requires_grad:
            full_grad[1] = x.value ** y.value * np.log(x.value)
        return full_grad


# Single variable operations:
class Negation(Operation):
    def __init__(self):
        super(Negation, self).__init__()
        self.name = '-'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(-x.value, _children=(x,), _op=Negation())

    def df(self, x):
        super().df(x)
        return [-1]


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


class Exp(Operation):
    def __init__(self):
        super(Exp, self).__init__()
        self.name = 'exp'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(np.exp(x.value), _children=(x,), _op=Exp())

    def df(self, x):
        super().df(x)
        return [np.exp(x.value)]


class Log(Operation):
    def __init__(self):
        super(Log, self).__init__()
        self.name = 'log'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(np.log(x.value), _children=(x,), _op=Log())

    def df(self, x):
        super().df(x)
        return [1 / x.value]


class Tanh(Operation):
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = 'tanh'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(np.tanh(x.value), _children=(x,), _op=Tanh())

    def df(self, x):
        super().df(x)
        return [1 - np.tanh(x.value) ** 2]


class Sin(Operation):
    def __init__(self):
        super(Sin, self).__init__()
        self.name = 'sin'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(np.sin(x.value), _children=(x,), _op=Sin())

    def df(self, x):
        super().df(x)
        return [np.cos(x.value)]


class Cos(Operation):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'
        self.single_variable_op = True

    def f(self, x):
        super().f(x)
        return Variable(np.cos(x.value), _children=(x,), _op=Cos())

    def df(self, x):
        super().df(x)
        return [-np.sin(x.value)]



if __name__ == '__main__':
    from optimizer import GD
    import matplotlib.pyplot as plt


    def z_func(x, y):
        # return (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)
        # return x**2 + y**2
        return - x ** 2 + y ** 2


    class MulVariableMod(Module):
        def __init__(self):
            super(MulVariableMod, self).__init__()
            self.x = Variable(0.5)
            self.y = Variable(1)
            print(f"Starting initialization: x={self.x.value}, y={self.y.value}")
            self.params.append(self.x)
            self.params.append(self.y)

        def forward(self):
            # return (1-(self.x**2+self.y**3))*exp(-(self.x**2+self.y**2)/2)
            return z_func(self.x, self.y)


    def perform_2d_gd(module, optim, epochs=20):
        func_value_history = []
        x_value_history = []
        y_value_history = []

        for epoch in range(epochs):
            x_value_history.append(module.x.value)
            y_value_history.append(module.y.value)

            optim.zero_grad()
            func_value = module.forward()
            func_value.backward()
            assert (math.isclose(func_value.value, z_func(module.x.value, module.y.value), abs_tol=1e-2))
            optim.step()

            func_value_history.append(func_value.value)
            if epoch % 2 == 0:
                print(f"epoch: {epoch} | x: {module.x.value} | y: {module.y.value} | f(x): {func_value.value}")
        return np.array(x_value_history), np.array(y_value_history), np.array(func_value_history)

    module = MulVariableMod()
    optim = GD(params=module.params, lr=0.1)
    x_value_history, y_value_history, func_value_history = perform_2d_gd(module, optim)

    x = np.arange(-3.0, 3.0, 0.1)
    y = np.arange(-3.0, 3.0, 0.1)
    [X, Y] = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1)
    Z = z_func(X, Y)
    ax.contourf(X, Y, Z)
    ax.set_title('Filled Contour Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.plot(x_value_history, y_value_history, 'o')

    plt.show()