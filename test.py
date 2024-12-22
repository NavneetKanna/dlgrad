from dlgrad.tensor import Tensor

a = Tensor.rand((4, 2, 3), requires_grad=True)
b = Tensor.rand((4, 2, 3), requires_grad=False)

c = a+b

c.sum().backward()

# print(a.grad.numpy())
# print(b.grad.numpy())

