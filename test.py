from dlgrad import Tensor
# from dlgrad import nn

# a = nn.Linear(3, 4)
# b = Tensor.rand((2, 3))

# c = a(b)

# print(c)

a = Tensor.rand((2))
print(a)
b = Tensor.rand((3, 2))
a+b
