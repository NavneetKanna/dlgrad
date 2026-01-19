# ruff: noqa
import json
import struct
import os
import mmap

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

def load_model(model, path: str):
    """
    Loads weights from .safetensors directly into the model's existing Tensors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")

    # 1. Map the current model structure
    # We need this to know WHERE to put the data we read
    current_state = get_state_dict(model)

    with open(path, 'rb') as f:
        # Read Header Size
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]

        # Read Header
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size

        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            loaded_count = 0

            for name, meta in header.items():
                if name not in current_state:
                    print(f"Warning: {name} found in file but not in model. Skipping.")
                    continue

                target_tensor = current_state[name]

                if tuple(target_tensor.data.metadata.shape) != tuple(meta['shape']):
                    print(f"Shape mismatch for {name}: Model {target_tensor.data.metadata.shape} vs File {meta['shape']}")
                    continue

                # Extract bytes
                start, end = meta['data_offsets']
                data_bytes = mm[data_start + start : data_start + end]

                # Write directly to C memory
                target_tensor.copy_from(data_bytes)
                loaded_count += 1

    print(f"Successfully loaded {loaded_count} tensors.")
