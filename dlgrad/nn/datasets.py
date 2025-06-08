from dlgrad import Tensor
from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType
from dlgrad.helpers import CACHE_DIR, fetch, unzip
from dlgrad.runtime.cpu import CPU


def mnist(device: Device = Device.CPU) -> list[Tensor]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    datasets = [
        ("train-images-idx3-ubyte.gz", True, 2051, (60000, 784)),
        ("train-labels-idx1-ubyte.gz", False, 2049, (60000, 1)),
        ("t10k-images-idx3-ubyte.gz", True, 2051, (10000, 784)),
        ("t10k-labels-idx1-ubyte.gz", False, 2049, (10000, 1)),
    ]

    tensors: list[Tensor] = []

    for filename, is_image, magic_number, shape in datasets:
        file_url = f"{base_url}{filename}"
        save_filename = filename.replace('.gz', '')

        fetch(url=file_url, filename=filename)
        unzip(filename=filename, save_filename=save_filename)

        data = CPU.mnist_loader(images=is_image, path=f"{CACHE_DIR}/downloads/{save_filename}", magic_number=magic_number)
        tensor = Tensor(
            data=Buffer(data=data, shape=shape, dtype=DType.FLOAT32, device=device),
            requires_grad=False, device=device
        )
        tensors.append(tensor)

    return tensors
