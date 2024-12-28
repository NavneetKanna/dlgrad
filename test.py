from dlgrad.tensor import Tensor
import numpy as np

np_data = np.random.uniform(low=-1.0, high=1.0, size=(2, 3)).astype(np.float32)
a = Tensor(np_data, requires_grad=True)
b = a.relu()

b.sum().backward()

print(a.numpy())
print(b.numpy())
print(a.grad.numpy())

