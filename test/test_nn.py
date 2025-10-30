import numpy as np
import torch
from dlgrad import Tensor, nn


def test_linear():
    BS, in_dim, out_dim = 32, 8, 16

    for d in ["cpu", "metal"]:
        dlgrad_model = nn.Linear(in_dim, out_dim, device=d)
        dlgrad_inp = Tensor.rand((BS, in_dim), device=d)
        dlgrad_out = dlgrad_model(dlgrad_inp)

        with torch.no_grad():
            if d == "metal":
                d = "mps"
            torch_model = torch.nn.Linear(in_dim, out_dim, device=d)
            torch_model.weight[:] = torch.tensor(dlgrad_model.weight.numpy(), device=d, dtype=torch.float32)
            torch_model.bias[:] = torch.tensor(dlgrad_model.bias.numpy(), device=d, dtype=torch.float32)

            torch_inp = torch.tensor(dlgrad_inp.numpy(), device=d)

            torch_out = torch_model(torch_inp)

        if d == "mps":
            torch_out = torch_out.cpu()

        np.testing.assert_allclose(dlgrad_out.numpy(), torch_out.numpy(), atol=5e-4, rtol=1e-5)
