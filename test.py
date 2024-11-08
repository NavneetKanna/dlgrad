from dlgrad import Tensor

a = Tensor.rand((2, 3))

print(a.numpy())

b = a.T

print(b.numpy())
