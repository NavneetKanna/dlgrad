# ruff: noqa
import json
import struct

from dlgrad import Tensor


def get_state_dict(obj, prefix: str = "") -> dict[str, Tensor]:
    """
    Recursively finds all Tensors and returns a dict: {'path.to.tensor': Tensor}
    """
    state_dict = {}

    def extract(o, current_prefix):
        if isinstance(o, Tensor):
            state_dict[current_prefix] = o
        elif isinstance(o, dict):
            for k, v in o.items():
                extract(v, f"{current_prefix}.{k}" if current_prefix else f"{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                extract(v, f"{current_prefix}.{i}" if current_prefix else f"{i}")
        elif hasattr(o, "__dict__"):
            for k, v in o.__dict__.items():
                if k.startswith("_"):
                    continue
                extract(v, f"{current_prefix}.{k}" if current_prefix else f"{k}")

    extract(obj, prefix)

    return state_dict

def get_parameters(obj) -> list[Tensor]:  # noqa: ANN001
    state_dict = []

    def extract_tensors(o):  # noqa: ANN001, ANN202
        if isinstance(o, Tensor):
            state_dict.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                extract_tensors(v)
        elif isinstance(o, list | tuple):
            for v in o:
                extract_tensors(v)
        elif hasattr(o, "__dict__"):
            extract_tensors(o.__dict__)

    extract_tensors(obj)

    return state_dict

def save_model(model, path: str):
    """
    Saves a model to .safetensors format.
    """
    state_dict = get_state_dict(model)

    header = {}
    offset = 0
    raw_data_chunks = []

    for name, tensor in state_dict.items():
        data_bytes = tensor.tobytes()
        length = len(data_bytes)

        header[name] = {
            "dtype": "F32",
            "shape": tensor.data.metadata.shape,
            "data_offsets": [offset, offset + length]
        }

        raw_data_chunks.append(data_bytes)
        offset += length

    json_header = json.dumps(header).encode('utf-8')

    with open(path, 'wb') as f:
        # Header Size (8 bytes uint64)
        f.write(struct.pack('<Q', len(json_header)))
        # Header JSON
        f.write(json_header)
        # Data Blobs
        for chunk in raw_data_chunks:
            f.write(chunk)

    print(f"Saved {len(state_dict)} tensors to {path}")
