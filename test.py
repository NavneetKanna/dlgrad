from dlgrad.device import Device
from dlgrad.tensor import Tensor

a = Tensor(1, device="cpu")
print(a.device)
