from dlgrad import Tensor


a = Tensor.rand((2, 3))

b = -a

print(a.numpy())
print(b.numpy())