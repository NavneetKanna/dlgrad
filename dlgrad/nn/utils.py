from dlgrad import Tensor


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
