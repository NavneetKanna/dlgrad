from dlgrad.tensor import Tensor


class Optimiser:
    def __init__(self, params: list[Tensor], lr) -> None:
        self.params = params
        self.lr = lr

    def step(self):
        for i in self.params:
            assert i.weight.shape == i.grad.shape, f"The shape of the weight {i.weight.shape} and its grad {i.grad.shape} do not match"
            i.weight = i.weight - self.lr*i.grad
            i.bias = i.bias - self.lr*i.bias

    def zero_grad(self):
        for i in self.params:
            i.weight.grad = None
            i.bias.grad = None


class SDG(Optimiser):
    def __init__(self, params: list[Tensor], lr=1e-3) -> None:
        super().__init__(params, lr)
