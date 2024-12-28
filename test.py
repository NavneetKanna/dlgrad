from dlgrad.tensor import Tensor

a = Tensor.rand((2, 3))

b = a>0.5

print(a.numpy())
print(b.numpy())