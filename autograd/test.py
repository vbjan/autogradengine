import autograd
from autograd import Variable, Module, Sigmoid
import torch


class TestModule1(Module):
    def __init__(self, var_constructor, use_torch=False):
        super(TestModule1, self).__init__(var_constructor, use_torch)
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
    def __init__(self, var_constructor, use_torch=False):
        super(TestModule2, self).__init__(var_constructor, use_torch)
        if self.use_torch:
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
    def __init__(self, var_constructor, use_torch=False):
        super(TestModule3, self).__init__(var_constructor, use_torch)
        if use_torch:
            self.exp = torch.exp
        else:
            self.exp = autograd.Exp()
        self.x = var_constructor(-2.)
        self.y = var_constructor(2.)
        self.params = self.collect_parameters()

    def forward(self):
        # return (1-(self.x**2+self.y**3))*exp(-(self.x**2+self.y**2)/2)
        return 1. - (self.x**2+self.y**3.)*self.exp(-(self.x**2.+self.y**2.)/2.)


def create_torch_tensor(x):
    a = torch.Tensor([x])
    a.requires_grad = True
    a.retain_grad()
    return a


def check_gradients(module):
    func = module(Variable, use_torch=False)
    output = func.forward()
    output.backward()

    correct_func = module(create_torch_tensor, use_torch=True)
    correct_output = correct_func.forward()
    correct_output.backward()
    print(f"\nengine output: {output}, \npytorch output: {correct_output}")
    print(f"engine gradients: {func} \npytorch gradients: {correct_func} ")
    if func == correct_func:
        print("CORRECT gradients!")
    else:
        print("FAIL: Incorrect gradients!")


if __name__ == '__main__':
    check_gradients(TestModule1)
    check_gradients(TestModule2)
    check_gradients(TestModule3)