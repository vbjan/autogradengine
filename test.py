from autograd import Variable, Module, Sigmoid
import torch


class TestModule1(Module):
    def __init__(self, var, use_torch=False):
        super(TestModule1, self).__init__(var, use_torch)
        self.a = var(2)
        self.b = var(-3)
        self.c = var(10)
        self.params = [self.a, self.b, self.c]

    def forward(self):
        e = self.a ** self.b
        f = self.c * self.b
        g = e + f
        g = g * g
        return g


class TestModule2(Module):
    def __init__(self, var, use_torch=False):
        super(TestModule2, self).__init__(var, use_torch)
        if self.use_torch:
            self.sig = torch.nn.Sigmoid()
        else:
            self.sig = Sigmoid()
        self.a = var(2)
        self.b = var(1)
        self.c = var(-3)
        self.params = [self.a, self.b, self.c]

    def forward(self):
        d = self.a + self.b - self.c
        d = self.sig(d ** 2)
        f = self.a * self.b
        g = d + f
        return g


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
    print(f"\n engine gradients: {func} \npytorch gradients: {correct_func} ")
    if func == correct_func:
        print("Correct gradients!")
    else:
        print("FAIL: Incorrect gradients!")


if __name__ == '__main__':
    check_gradients(TestModule1)
    check_gradients(TestModule2)