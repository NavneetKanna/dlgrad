import numpy as np
import torch
import pytest
from dlgrad import Tensor, nn

@pytest.fixture(params=["cpu", "metal"])
def device(request):
    if request.param == "metal" and not torch.backends.mps.is_available():
        pytest.skip("Metal not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

def test_linear(device):
    BS, in_dim, out_dim = 32, 8, 16

    dlgrad_model = nn.Linear(in_dim, out_dim, device=device)
    dlgrad_inp = Tensor.rand((BS, in_dim), device=device)
    dlgrad_out = dlgrad_model(dlgrad_inp)

    with torch.no_grad():
        torch_model = torch.nn.Linear(in_dim, out_dim, device=to_torch_device(device))
        torch_model.weight[:] = torch.tensor(dlgrad_model.weight.numpy(), device=to_torch_device(device), dtype=torch.float32)
        torch_model.bias[:] = torch.tensor(dlgrad_model.bias.numpy(), device=to_torch_device(device), dtype=torch.float32)

        torch_inp = torch.tensor(dlgrad_inp.numpy(), device=to_torch_device(device))
        torch_out = torch_model(torch_inp)

    np.testing.assert_allclose(dlgrad_out.numpy(), torch_out.cpu().numpy(), atol=5e-4, rtol=1e-5)

def test_embedding_forward_backward(device):
    vocab_size = 10
    embed_dim = 4
    batch_size = 2
    seq_len = 3

    indices_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)

    dl_emb = nn.Embedding(vocab_size, embed_dim)
    dl_input = Tensor(indices_np.astype(np.float32), device=device)

    dl_out = dl_emb(dl_input)

    grad_np = np.random.randn(*dl_out.shape).astype(np.float32)
    dl_grad = Tensor(grad_np, device=device)

    (dl_out * dl_grad).sum().backward()

    torch_device = to_torch_device(device)
    torch_input = torch.tensor(indices_np, dtype=torch.long, device=torch_device)
    torch_emb = torch.nn.Embedding(vocab_size, embed_dim).to(torch_device)

    with torch.no_grad():
        torch_emb.weight.copy_(torch.from_numpy(dl_emb.weight.numpy()))

    torch_emb.weight.requires_grad = True

    torch_out = torch_emb(torch_input)
    torch_grad = torch.tensor(grad_np, device=torch_device)

    torch_out.backward(torch_grad)

    np.testing.assert_allclose(dl_out.numpy(), torch_out.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

    dl_weight_grad = dl_emb.weight.grad.numpy()
    torch_weight_grad = torch_emb.weight.grad.detach().cpu().numpy()

    np.testing.assert_allclose(dl_weight_grad, torch_weight_grad, atol=1e-5, rtol=1e-5)

