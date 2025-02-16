
from dlgrad.tensor import Tensor


class Optimiser:
    def __init__(self, params: list[Tensor], lr: int = 1e-1) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for i in self.params:
            assert i.shape == i.grad.shape, f"The shape of the weight {i.shape} and its grad {i.grad.shape} do not match"

            # print("i before")
            # print(i.numpy())
            # print("grad")
            # print(i.grad.numpy())
            # print("lr")
            # print(self.lr)
            t = i.grad*self.lr
            # print("t")
            # print(t.numpy())
            i.data = (i - t).data
            # print("i after")
            # print(i.numpy())

    def zero_grad(self) -> None:
        for i in self.params:
            i.grad = None


class SGD(Optimiser):
    def __init__(self, params: list[Tensor], lr: int = 1e-3) -> None:
        super().__init__(params, lr)
