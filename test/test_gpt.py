import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dlgrad import Tensor, nn
from itertools import product

# --- Fixtures & Helpers ---

@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if request.param == 'metal' and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

def check_val(dl_tensor, torch_tensor, atol=1e-5, rtol=1e-3):
    """Helper to compare tensor values"""
    np.testing.assert_allclose(
        dl_tensor.numpy(), 
        torch_tensor.detach().cpu().numpy(), 
        atol=atol, rtol=rtol
    )

def check_grad(dl_args, torch_args, dl_out, torch_out, atol=1e-4, rtol=1e-3):
    """
    Helper to run backward on a scalar sum and compare input gradients.
    """
    # 1. Backprop from a simple scalar signal
    dl_out.sum().backward()
    torch_out.sum().backward()

    # 2. Compare gradients of inputs
    for dl_x, torch_x in zip(dl_args, torch_args):
        if dl_x.requires_grad:
            assert torch_x.grad is not None, f"Torch grad missing for {torch_x}"
            assert dl_x.grad is not None, f"Dlgrad grad missing for {dl_x}"
            check_val(dl_x.grad, torch_x.grad, atol=atol, rtol=rtol)

# --- Core Ops Tests ---

@pytest.mark.parametrize("shape", [
    (32, 32),           # Square
    (8, 16, 32),        # 3D Batch
    (4, 8, 16, 32),     # 4D (Attention-like)
    (1, 1),             # Scalar-like edge case
])
def test_matmul(shape, device):
    # Create random data
    B_dims = shape[:-2]
    M, K = shape[-2], shape[-1]
    N = 16 # arbitrary output dim

    # Input 1: (..., M, K)
    np_x = np.random.randn(*shape).astype(np.float32)
    # Input 2: (..., K, N) - PyTorch linear is (N, K) but matmul is (K, N)
    np_y = np.random.randn(*(B_dims + (K, N))).astype(np.float32)

    # Setup Tensors
    x = Tensor(np_x, device=device, requires_grad=True)
    y = Tensor(np_y, device=device, requires_grad=True)

    tx = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)
    ty = torch.tensor(np_y, device=to_torch_device(device), requires_grad=True)

    # Forward
    z = x @ y
    tz = tx @ ty

    check_val(z, tz)
    check_grad([x, y], [tx, ty], z, tz)

@pytest.mark.parametrize("shape", [(2, 3, 4)])
def test_transpose_reshape(shape, device):
    np_x = np.random.randn(*shape).astype(np.float32)
    x = Tensor(np_x, device=device, requires_grad=True)
    tx = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)

    # Operation: Transpose(1, 2) -> Reshape
    z = x.transpose(1, 2).reshape((shape[0], shape[2] * shape[1]))
    tz = tx.transpose(1, 2).reshape(shape[0], shape[2] * shape[1])

    check_val(z, tz)
    check_grad([x], [tx], z, tz)

@pytest.mark.parametrize("shape", [(4, 128, 256)]) # (B, T, C)
def test_rmsnorm(shape, device):
    np_x = np.random.randn(*shape).astype(np.float32)
    dim = shape[-1]

    # Dlgrad
    x = Tensor(np_x, device=device, requires_grad=True)
    rms = nn.RMSNorm(dim)
    z = rms(x)

    # Torch (Manual RMSNorm implementation)
    tx = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)
    tweight = torch.tensor(rms.weight.numpy(), device=to_torch_device(device), requires_grad=True)

    # Torch RMSNorm formula
    tvar = tx.pow(2).mean(-1, keepdim=True)
    tnorm = tx * torch.rsqrt(tvar + rms.eps)
    tz = tnorm * tweight

    check_val(z, tz)

    # We cheat slightly: Copy dlgrad weights to torch to ensure grad check works for input x
    # We can also check gradients for the weight if needed
    z.sum().backward()
    tz.sum().backward()

    check_val(x.grad, tx.grad)
    check_val(rms.weight.grad, tweight.grad)

@pytest.mark.parametrize("vocab_size", [100])
@pytest.mark.parametrize("embed_dim", [32])
def test_embedding(vocab_size, embed_dim, device):
    # Indices are integers, no grad
    idx_shape = (4, 16) # (B, T)
    np_idx = np.random.randint(0, vocab_size, size=idx_shape).astype(np.int32)

    # Dlgrad
    emb = nn.Embedding(vocab_size, embed_dim)
    # Important: Set Embedding weight to known values for comparison
    np_w = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    emb.weight = Tensor(np_w, device=device, requires_grad=True)

    x = Tensor(np_idx.astype(np.float32), device=device) # dlgrad often uses float indices
    z = emb(x)

    # Torch
    tx = torch.tensor(np_idx, device=to_torch_device(device)).long()
    temb = torch.nn.Embedding(vocab_size, embed_dim)
    temb.weight = torch.nn.Parameter(torch.tensor(np_w, device=to_torch_device(device)))

    tz = temb(tx)

    check_val(z, tz)

    z.sum().backward()
    tz.sum().backward()

    # Check gradients on the WEIGHTS, not inputs (inputs are indices)
    check_val(emb.weight.grad, temb.weight.grad)

def test_cross_entropy(device):
    # Logits: (B*T, Vocab)
    B, T, V = 4, 32, 100
    shape = (B*T, V)

    np_logits = np.random.randn(*shape).astype(np.float32)
    # Targets: (B*T,) indices
    np_targets = np.random.randint(0, V, size=(B*T,)).astype(np.int32)

    # Dlgrad
    logits = Tensor(np_logits, device=device, requires_grad=True)
    # Assuming dlgrad takes targets as (N, 1) or flat? Adjust based on your engine.
    targets = Tensor(np_targets.reshape(-1, 1).astype(np.float32), device=device) 

    loss = logits.cross_entropy_loss(targets)

    # Torch
    tlogits = torch.tensor(np_logits, device=to_torch_device(device), requires_grad=True)
    ttargets = torch.tensor(np_targets, device=to_torch_device(device)).long()

    tloss = F.cross_entropy(tlogits, ttargets)

    check_val(loss, tloss)

    loss.backward()
    tloss.backward()

    check_val(logits.grad, tlogits.grad)

@pytest.mark.parametrize("seq_len", [32, 128])
def test_causal_mask(seq_len, device):
    """Test the specific masking logic used in CausalSelfAttention"""
    # Create dummy attention scores (B, H, T, T)
    shape = (2, 4, seq_len, seq_len)
    np_att = np.random.randn(*shape).astype(np.float32)

    x = Tensor(np_att, device=device, requires_grad=True)
    tx = torch.tensor(np_att, device=to_torch_device(device), requires_grad=True)

    # 1. Create Mask
    mask = Tensor.tril(Tensor.ones((seq_len, seq_len), device=device), k=0.0)
    tmask = torch.tril(torch.ones((seq_len, seq_len), device=to_torch_device(device)), diagonal=0)

    # 2. Masked Fill (Causal)
    # Using the safe -1e9 value we discussed
    fill_val = -1e9

    y = x.masked_fill(mask == Tensor(0.0, device=device), fill_val)

    # Torch equivalence
    ty = tx.masked_fill(tmask == 0, fill_val)

    # 3. Softmax
    z = y.softmax(dim=-1)
    tz = ty.softmax(dim=-1)

    check_val(z, tz)
    check_grad([x], [tx], z, tz)
