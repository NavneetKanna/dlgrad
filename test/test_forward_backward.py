import numpy as np
import pytest
import torch
from dlgrad import Tensor, nn
import platform

BS, in_dim, HS, ncls = 128, 784, 64, 10

device_pairs = [
    ('cpu', 'cpu'),
]

if platform.system() == 'Darwin':
    device_pairs.append(('metal', 'mps'))

def run_test(dlgrad_device, torch_device):
    na = np.random.uniform(size=(BS, in_dim)).astype(np.float32)
    nt = np.random.randint(0, ncls, size=(BS, 1)).astype(np.float32)  # randint(0, ncls) gives 0 to 3 for ncls=4

    dlgrad_inp = Tensor(na)  # No .to() since not implemented yet
    to_inp = torch.tensor(na).to(torch_device)

    dlgrad_target = Tensor(nt)
    to_target = torch.tensor(nt, dtype=torch.long).squeeze(-1).to(torch_device)  # squeeze last dim to make 1D (BS,)

    # dlgrad model (assumes created on default device, e.g., CPU)
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
    tli1 = torch.nn.Linear(in_dim, HS).to(torch_device)
    tli2 = torch.nn.Linear(HS, ncls).to(torch_device)
    with torch.no_grad():
        # Weights are (out, in), should be 2D and match directly
        tli1.weight.data.copy_(torch.from_numpy(li1.weight.numpy()).to(torch_device))
        tli2.weight.data.copy_(torch.from_numpy(li2.weight.numpy()).to(torch_device))
        # Biases in dlgrad are likely (1, out_features), while torch expects (out_features,); squeeze dlgrad bias
        tli1.bias.data.copy_(torch.from_numpy(li1.bias.numpy().squeeze(0)).to(torch_device))
        tli2.bias.data.copy_(torch.from_numpy(li2.bias.numpy().squeeze(0)).to(torch_device))
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

    # Check forward before backward
    np.testing.assert_allclose(dl_out.numpy(), to_out.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(dl_loss.numpy(), torch_loss.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)

    # backward
    dl_loss.backward()
    torch_loss.backward()

    # Check gradients match (before step); reshape torch grads to match dlgrad's (1, out_features) shape if needed
    np.testing.assert_allclose(li1.weight.grad.numpy(), tli1.weight.grad.cpu().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.weight.grad.numpy(), tli2.weight.grad.cpu().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li1.bias.grad.numpy(), tli1.bias.grad.cpu().numpy().reshape(1, -1), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.bias.grad.numpy(), tli2.bias.grad.cpu().numpy().reshape(1, -1), atol=1e-3, rtol=1e-3)

    # step
    dl_optimizer = nn.optim.SGD(nn.utils.get_parameters(m), lr=1e-3)
    to_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    dl_optimizer.step()
    to_optimizer.step()

    # Clear grads to avoid accumulation if needed
    dl_optimizer.zero_grad()
    to_optimizer.zero_grad()

    # assertions after step; reshape torch biases to match dlgrad's (1, out_features) shape
    np.testing.assert_allclose(li1.weight.numpy(), tli1.weight.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.weight.numpy(), tli2.weight.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li1.bias.numpy(), tli1.bias.detach().cpu().numpy().reshape(1, -1), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(li2.bias.numpy(), tli2.bias.detach().cpu().numpy().reshape(1, -1), atol=1e-3, rtol=1e-3)

    print(f"Test passed for dlgrad_device='{dlgrad_device}', torch_device='{torch_device}'")

def test():
    for dlgrad_dev, torch_dev in device_pairs:
        run_test(dlgrad_dev, torch_dev)

if __name__ == "__main__":
    test()
