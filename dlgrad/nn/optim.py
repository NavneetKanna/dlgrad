from dlgrad.tensor import Tensor


class Optimiser:
    def __init__(self, params: list[Tensor], lr: int = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for i in self.params:
            assert i.shape == i.grad.shape, f"The shape of the weight {i.shape} and its grad {i.grad.shape} do not match"
            # if i.shape == (16, 784):
            #     print("i before")
            #     print(id(i))
            #     print("i.grad")
            #     print(i.grad.numpy().any())
            i.data = (i - i.grad*self.lr).data
            # if i.shape == (16, 784):
            #     print("init wt after updated")
            #     print(id(i))
            # print(id(i))
                # print(i.grad.numpy())

    def zero_grad(self) -> None:
        for i in self.params:
            i.grad = None


class SDG(Optimiser):
    def __init__(self, params: list[Tensor], lr: int = 1e-3) -> None:
        super().__init__(params, lr)
