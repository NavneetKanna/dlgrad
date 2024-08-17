

from typing import Any


class Module:
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> Any:
        return self.forward(x)