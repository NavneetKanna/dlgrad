from dlgrad import Tensor
from dlgrad import nn
import numpy as np
import torch



def test_linear():
    BS, in_dim, out_dim = 32, 8, 16

    dlgrad_model = nn.Linear(in_dim, out_dim)
    dlgrad_inp = Tensor.rand((BS, in_dim))
    dlgrad_out = dlgrad_model(dlgrad_inp)

    with torch.no_grad():
        torch_model = torch.nn.Linear(in_dim, out_dim)
        torch_model.weight[:] = torch.tensor(dlgrad_model.weight.numpy(), dtype=torch.float32)
        torch_model.bias[:] = torch.tensor(dlgrad_model.bias.numpy(), dtype=torch.float32)

        torch_inp = torch.tensor(dlgrad_inp.numpy())

        torch_out = torch_model(torch_inp)

    np.testing.assert_allclose(dlgrad_out.numpy(), torch_out.numpy(), atol=5e-4, rtol=1e-5)
