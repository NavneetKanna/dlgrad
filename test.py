from dlgrad.tensor import Tensor

a = Tensor.rand(2, 3)
a.numpy()
b = Tensor.transpose(a)
b.numpy()