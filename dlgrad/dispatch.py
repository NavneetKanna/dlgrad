# from dlgrad.tensor import Tensor


class Dispatcher:
    def __init__(self) -> None:
        self._dispatch_table: dict = {}

    def register(self, op, device):
        def decorator(func):
            self._dispatch_table[(op, device)] = func
        return decorator

    def dispatch(self, x, op, **kwargs):
        device = kwargs["device"]
        return self._dispatch_table[(op, device)](x)

dispatcher = Dispatcher()