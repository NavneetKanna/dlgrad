from dlgrad import Tensor
from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType
from dlgrad.helpers import CACHE_DIR, fetch, unzip
from dlgrad.runtime.cpu import CPU


def mnist() -> list[Tensor]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    t = [
        ("train-images-idx3-ubyte.gz", True, 2051, (60000, 784)),
        ("train-labels-idx1-ubyte.gz", False, 2049, (60000, 1)),
        ("t10k-images-idx3-ubyte.gz", True, 2051, (10000, 784)),
        ("t10k-labels-idx1-ubyte.gz", False, 2049, (10000, 1)),
    ]
    res: list[Tensor] = []
    for u in t:
        fetch(url=base_url+u[0], filename=u)
        unzip(path=f"{CACHE_DIR}/downloads/{u[0]}", save_path=f"{CACHE_DIR}/downloads/{''.join(u[0].split('.')[:-1])}")
        data = CPU.mnist_loader(images=u[1], path=f"{CACHE_DIR}/downloads/{''.join(u[0].split('.')[:-1])}", magic_number=u[2])  # noqa: E501
        res.append(
            Tensor(
                data=Buffer(data=data, shape=u[3], dtype=DType.FLOAT32, device=Device.CPU),
                requires_grad=False
            )
        )

    return res
