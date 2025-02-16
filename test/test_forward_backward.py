import numpy as np
import pytest
import torch

from dlgrad import Tensor, nn

BS, in_dim, HS, ncls = 2, 3, 5, 4

na = np.random.uniform(size=(BS, in_dim)).astype(np.float32)
nt = np.random.randint(0, ncls-1, size=(BS, 1)).astype(np.float32)


dlgrad_inp = Tensor(na)
to_inp = torch.tensor(na)

dlgrad_target = Tensor(nt)
to_target = torch.tensor(nt, dtype=torch.long).squeeze()

# dlgrad model

li1 = nn.Linear(in_dim, HS)
li2 = nn.Linear(HS, ncls)
class Model:
    def __init__(self):
        self.layers = [
            li1,
            Tensor.relu,
            li2
        ]
    
    def __call__(self, x: Tensor) -> Tensor: 
        return x.sequential(self.layers)

m = Model()
dl_out = m(dlgrad_inp)

# torch model

with torch.no_grad():
    tli1 = torch.nn.Linear(in_dim, HS)
    tli1.weight[:] = torch.tensor(li1.weight.numpy(), dtype=torch.float32)
    tli1.bias[:] = torch.tensor(li1.bias.numpy(), dtype=torch.float32)

    tli2 = torch.nn.Linear(HS, ncls)
    tli2.weight[:] = torch.tensor(li2.weight.numpy(), dtype=torch.float32)
    tli2.bias[:] = torch.tensor(li2.bias.numpy(), dtype=torch.float32)
model = torch.nn.Sequential(
    tli1,
    torch.nn.ReLU(),
    tli2,
)

to_out = model(to_inp)

# loss

dl_loss = dl_out.cross_entropy_loss(dlgrad_target)
l = torch.nn.CrossEntropyLoss(reduction="sum")
torch_loss = l(to_out, to_target)



# backward

dl_loss.backward()
torch_loss.backward()

# step

dl_optimizer = nn.optim.SGD(nn.utils.get_parameters(m))
to_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dl_optimizer.step()
to_optimizer.step()


def test():
    np.testing.assert_allclose(dl_out.numpy(), to_out.detach().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(dl_loss.numpy(), torch_loss.detach().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li1.weight.numpy(), tli1.weight.detach().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.weight.numpy(), tli2.weight.detach().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li1.bias.numpy(), tli1.bias.reshape(1, -1).detach().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.bias.numpy(), tli2.bias.reshape(1, -1).detach().numpy(), atol=1e-3, rtol=1e-3)
