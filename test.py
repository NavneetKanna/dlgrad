from dlgrad.tensor import Tensor

a = Tensor.rand((2, 3))
b = a.T

print(a.data.shape)
print(b.data.shape)
print(a.numpy())
print(b.numpy())
print(a.data.stride)
print(b.data.stride)
# b = Tensor.rand((2, 3))

# print(a.device)

# c = a+b

