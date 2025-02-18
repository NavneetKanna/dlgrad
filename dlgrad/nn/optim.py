
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

class Adam(Optimiser):
    def __init__(self, params: list[Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:  # noqa: E501
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {id(p): Tensor.zeros_like(p.shape) for p in params}
        self.v = {id(p): Tensor.zeros_like(p.shape) for p in params}
        self.t = 0
        self.params = params

    def step(self):  # noqa: ANN201
        self.t += 1
        for p in self.params:
            pid = id(p)
            g = p.grad
            self.m[pid] = (self.m[pid] * self.beta1) + (g * (1 - self.beta1))
            self.v[pid] = (self.v[pid] * self.beta2) + ((g ** 2) * (1 - self.beta2))

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            p.data = (p - (m_hat / (v_hat.sqrt() + self.eps)) * self.lr).data
