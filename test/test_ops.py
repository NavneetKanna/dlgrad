import numpy as np
import torch

from dlgrad import Tensor


# Thanks to tinygrad for the template
def run(shapes: list[tuple], func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data) for data in np_data]
    torch_data = [torch.tensor(data) for data in np_data]

    np.testing.assert_allclose(func(*dlgrad_data).numpy(), func(*torch_data).numpy(), atol=1e-6, rtol=1e-3)
    
def test_add_same_shape():
    sh1 = (2, 3)
    sh2 = (2, 3)
    run([sh1, sh2], lambda x, y: x+y)
    sh1 = (100, 100)
    sh2 = (100, 100)
    run([sh1, sh2], lambda x, y: x+y)
    sh1 = (78, 91)
    sh2 = (78, 91)
    run([sh1, sh2], lambda x, y: x+y)
