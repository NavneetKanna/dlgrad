import urllib.request
import gzip
import shutil
from dlgrad.tensor import Tensor


def mnist():
    x_train = Tensor.rand((10000, 784))
    y_train = Tensor.randint((10000,))
    # x_train = Tensor.rand((10000, 784))
    # x_train = Tensor.rand((10000, 784))

    # url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
    # file_name = url.split("/")[-1]

    # with urllib.request.urlopen(url) as response:
    #     with open(file_name, "wb") as f:
    #         f.write(response.read())

    # with gzip.open(file_name, "rb") as f_in, open(file_name[:-3], "wb") as f_out:
    #     shutil.copyfileobj(f_in, f_out)
