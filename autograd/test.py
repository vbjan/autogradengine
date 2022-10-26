import autograd
from autograd import Variable, Module, Sigmoid, Tanh
import torch
from autograd.nn import Linear
import numpy as np


def allocate_constructor(use_torch):
    if use_torch:
        return create_torch_tensor
    if not use_torch:
        return Variable


class TestModule1(Module):
    def __init__(self, use_torch):
        super(TestModule1, self).__init__(use_torch)
        self.use_torch = use_torch
        var_constructor = allocate_constructor(self.use_torch)
        self.a = var_constructor(2.)
        self.b = var_constructor(-3.)
        self.c = var_constructor(10.)
        self.params = self.collect_parameters()

    def forward(self):
        e = self.a ** self.b
        f = self.c * self.b
        g = e + f
        g = g * g
        return g


class TestModule2(Module):
    def __init__(self, use_torch):
        super(TestModule2, self).__init__(use_torch)
        self.use_torch = use_torch
        var_constructor = allocate_constructor(self.use_torch)
        if use_torch:
            self.sig = torch.nn.Sigmoid()
        else:
            self.sig = Sigmoid()
        self.a = var_constructor(2.)
        self.b = var_constructor(1.)
        self.c = var_constructor(-3.)
        self.params = self.collect_parameters()

    def forward(self):
        d = self.a + self.b - self.c
        d = self.sig(d ** 2.)
        f = self.a * self.b
        g = d + f
        return g


class TestModule3(Module):
    def __init__(self, use_torch):
        super(TestModule3, self).__init__(use_torch)
        self.use_torch = use_torch
        var_constructor = allocate_constructor(self.use_torch)
        if self.use_torch:
            self.exp = torch.exp
        else:
            self.exp = autograd.Exp()
        self.x = var_constructor(-2.)
        self.y = var_constructor(2.)
        self.params = self.collect_parameters()

    def forward(self):
        # return (1-(self.x**2+self.y**3))*exp(-(self.x**2+self.y**2)/2)
        return 1. - (self.x**2+self.y**3.)*self.exp(-(self.x**2.+self.y**2.)/2.)


class TorchMLP(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden):
        super(TorchMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, n_hidden, bias=True)  # initialialization is different!!!!
        self.linear2 = torch.nn.Linear(n_hidden, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()

    def forward(self, _x, _y_label):
        _y = self.tanh(self.linear2(self.relu(self.sigmoid(self.linear1(_x)))))
        return (torch.sum(_y) + (torch.ones(_x.shape, requires_grad=False)-_x).sum())**2


class TestModule4(Module):
    def __init__(self):
        super(TestModule4, self).__init__()
        input_size = 4
        output_size = 2
        n_hidden = 20
        self._x = np.random.rand(4)
        self._y_label = np.random.rand(2)

        # torch model
        self._torch_x = create_torch_tensor(self._x)
        self._torch_y_label = create_torch_tensor(self._y_label)
        self.TorchModel = TorchMLP(input_size, output_size, n_hidden=n_hidden)
        self.params = self.TorchModel.parameters()

        # autograd model
        self._autograd_x = Variable(self._x.reshape(-1, 1), requires_grad=False)
        self._autograd_y_label = Variable(self._y_label.reshape(-1, 1), requires_grad=False)
        self.linear1 = Linear((n_hidden, input_size))
        self.linear2 = Linear((output_size, n_hidden))

        # this is ugly
        self.linear1.weight = Variable(np.array(self.TorchModel.linear1.weight.detach()))
        self.linear2.weight = Variable(np.array(self.TorchModel.linear2.weight.detach()))
        self.linear1.bias = Variable(np.array(self.TorchModel.linear1.bias.detach()).reshape(-1, 1))
        self.linear2.bias = Variable(np.array(self.TorchModel.linear2.bias.detach()).reshape(-1, 1))
        self.linear1_weight = self.linear1.weight
        self.linear2_weight = self.linear2.weight
        self.linear1_bias = self.linear1.bias
        self.linear2_bias = self.linear2.bias

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.mseloss = autograd.MSELoss()
        self.relu = autograd.ReLu()

        if not np.allclose(self.linear1.weight.value, self.TorchModel.linear1.weight.detach().numpy()):
            raise AssertionError("Linear weight is not the same!")
        if not np.allclose(self.linear2.weight.value, self.TorchModel.linear2.weight.detach().numpy()):
            raise AssertionError("Linear weight is not the same!")

    def forward(self, use_torch):
        if use_torch:
            return self.TorchModel(self._torch_x, self._torch_y_label)
        else:
            _y = self.tanh(self.linear2(self.relu(self.sigmoid(self.linear1(self._autograd_x)))))
            return (_y.sum() + (Variable(np.ones(self._autograd_x.shape), requires_grad=False)-self._autograd_x).sum())**2


def create_torch_tensor(x):
    try:
        a = torch.Tensor(x)
    except:
        a = torch.Tensor([x])
    a.requires_grad = True
    a.retain_grad()
    return a


def check_gradients(module):
    func = module(use_torch=False)
    output = func.forward()
    output.backward()

    correct_func = module(use_torch=True)
    correct_output = correct_func.forward()
    correct_output.backward()
    print(f"\nengine output:\n{output}, \npytorch output:\n{correct_output}")
    print(f"engine gradients:\n{func} \npytorch gradients:\n{correct_func} ")
    if func == correct_func:
        print("CORRECT gradients!")
    else:
        print("FAIL: Incorrect gradients!")


def check_nn_gradients(Module):
    print("\nDoing NN test: ...")
    func = Module()

    output = func.forward(use_torch=False)
    output.backward()

    correct_output = func.forward(use_torch=True)
    correct_output.backward()

    if not np.allclose(output.value, correct_output.detach().numpy(), rtol=1e-5):
        raise AssertionError(f"Output is not the same! {output.value} != {correct_output.detach().numpy()} ")
    if np.allclose(func.linear1.weight.grad.transpose(), func.TorchModel.linear1.weight.grad.detach().numpy(), rtol=1e-5, atol=1e-5):
        #print("Componentwise erros: \n", func.linear1.weight.grad.transpose()-func.TorchModel.linear1.weight.grad.detach().numpy())
        print("CORRECT gradients!")
    else:
        print(func.linear1.weight.grad.transpose(), '\n', func.TorchModel.linear1.weight.grad.detach().numpy())
        print("FAIL: Incorrect gradient!")


if __name__ == '__main__':
    check_gradients(TestModule1)
    check_gradients(TestModule2)
    check_gradients(TestModule3)
    check_nn_gradients(TestModule4)