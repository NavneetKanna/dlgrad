from dlgrad import Tensor

a = Tensor.rand((2, 3), requires_grad=True)
b = Tensor.rand((2, 3), requires_grad=True)

c = a+b
# import code; code.interact(local=vars())
c.backward()
