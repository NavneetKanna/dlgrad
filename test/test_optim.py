from dlgrad import Tensor
from dlgrad.nn.optim import SGD, Adam
import torch
import numpy as np


np.random.seed(1337)

BS, in_dim, HS, ncls = 4, 784, 16, 10

# inp = np.random.uniform(size=(BS, in_dim)).astype(np.float32)
# w1 = np.random.uniform(size=(in_dim, HS)).astype(np.float32)


x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
b_init = np.random.randn(1,4).astype(np.float32)

class Model:
    def __init__(self, tensor):
        self.x = tensor(x_init.copy(), requires_grad=True)
        self.W = tensor(W_init.copy(), requires_grad=True)
    def forward(self):
        return (self.x * self.W).sum()

def step(tensor, optim, steps=2):
    net = Model(tensor)
    optim = optim([net.x, net.W], lr=1e-3)
    for _ in range(steps):
        out = net.forward()
        optim.zero_grad()
        out.backward()
        optim.step()
    
    if isinstance(net.x, torch.Tensor) and isinstance(net.W, torch.Tensor):
        return net.x.detach().numpy(), net.W.detach().numpy()
    else:
        return net.x.numpy(), net.W.numpy()


def _test_optim(dlgrad_optim, torch_optim, steps, atol, rtol):
    for x,y in zip(step(Tensor, dlgrad_optim, steps),
                   step(torch.tensor, torch_optim, steps)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
         
def test():
    _test_optim(SGD, torch.optim.SGD, 5, 1e-2, 1e-3)
    _test_optim(Adam, torch.optim.Adam, 5, 1e-2, 1e-3)
