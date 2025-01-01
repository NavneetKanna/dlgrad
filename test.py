# from dlgrad.tensor import Tensor
# import numpy as np

# # np_data = np.random.uniform(low=-1.0, high=1.0, size=(2, 3)).astype(np.float32)

# np_data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
# a = Tensor(np_data, requires_grad=True)
# np_data = np.arange(3).reshape(1, 3, 1).astype(np.float32)
# b = Tensor(np_data, requires_grad=True)

# c=a+b

# print(a.numpy())
# print(b.numpy())
# print(c.numpy())

import torch
import math
                    
def foo(shape1, shape2):
    a = torch.arange(math.prod(shape1)).reshape(shape1)
    b = torch.arange(math.prod(shape2)).reshape(shape2)
    c = a+b
    print("a", tuple(a.shape))
    print(a)
    print()
    print("b", tuple(b.shape))
    print(b)
    print()
    print("c", tuple(c.shape))
    print(c)


s1 = (2, 3, 4)
s2 = (1, 1, 1)
foo(s1, s2)
