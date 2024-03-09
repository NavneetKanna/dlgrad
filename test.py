from dlgrad import Tensor

a = Tensor.rand(2, 2)

print(a.numpy())
print(a[1][1])
