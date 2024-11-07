from dlgrad import Tensor

# x (3, 2) and y (2, 4)

a = Tensor.rand((2, 3))
b = Tensor.rand((3, 4))

a@b