from dlgrad.helpers import CACHE_DIR, fetch, unzip


def mnist() -> None:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    for u in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:  # noqa: E501
        fetch(url=base_url+u, filename=u)
        unzip(path=f"{CACHE_DIR}/downloads/{u}", save_path=f"{CACHE_DIR}/downloads/{''.join(u.split('.')[:-1])}")
