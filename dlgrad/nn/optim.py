from dlgrad.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: int = 1e-1) -> None:
        self.params = params
        self.lr = Tensor(lr)

    def step(self) -> None:
        for i in self.params:
            assert i.shape == i.grad.shape, f"The shape of the parameter {i.shape} and its grad {i.grad.shape} do not match"

            i.data = (i - i.grad*self.lr).data

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: int = 1e-3, momentum: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = {id(param): Tensor.zeros_like(param.shape) for param in self.params}

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue

            assert param.shape == param.grad.shape, (
                f"The shape of the parameter {param.shape} and its grad {param.grad.shape} do not match"
            )

            param_id = id(param)
            grad = param.grad

            if self.momentum != 0:
                velocity = self.velocities[param_id]
                velocity = self.momentum * velocity + grad
                self.velocities[param_id] = velocity
                param.data = (param - self.lr * velocity).data
            else:
                param.data = (param - self.lr * grad).data

# https://paperswithcode.com/method/adam
class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {id(p): Tensor.zeros_like(p.shape) for p in params}
        self.v = {id(p): Tensor.zeros_like(p.shape) for p in params}
        self.t = 0
        self.params = params
        self.tensor_one = Tensor(1.0)

    def step(self) -> None:
        self.t += 1
        for param in self.params:
            pid = id(param)
            g = param.grad
            self.m[pid] = (self.m[pid] * self.beta1) + (g * (1.0 - self.beta1))
            self.v[pid] = (self.v[pid] * self.beta2) + ((g ** 2.0) * (1.0 - self.beta2))
            m_hat = self.m[pid] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1.0 - self.beta2 ** self.t)
            param.data = (param - (m_hat / (v_hat.sqrt() + self.eps)) * self.lr).data
