from enum import Enum
from typing import Any

from dlgrad.buffer import Buffer


class Dispatcher:
    """
    Manages and dispatches operations to specific runtime implementations.

    This class maintains a dispatch table, which maps (op, device) to specific runtime implementations.

    Attributes:
        _dispatch_table (dict): A dictionary that maps a tuple of op and device to functions.
    """
    def __init__(self) -> None:
        self._dispatch_table: dict = {}

    def register(self, op: Enum, device: Enum):
        """
        Registers a function for a specific operation and device.

        This method is intended to be used as a decorator.

        Parameters:
            op (Enum) : Any of the op enums defined in the helper module.
            device (Enum) : Any of the device enums defined in the device module.
        """
        def decorator(func):
            self._dispatch_table[(op, device)] = func
        return decorator

    def dispatch(self, op: Enum, device: Enum, x: Any, y: Any = None, **kwargs) -> Buffer:
        """
        Calls the function registered in the dispatch table based on the op and device.

        Parameters:
            x (Any) : Data to be passed to the function.
            op (Enum) : Any of the op enums defined in the helper module. 
            device (Enum) : Any of the device enums defined in the device module.
            **kwargs (dict) : Any additional args.
        
        Returns:
            Buffer: A Buffer object.
        """
        args = [x]
        if y is not None:
            args.append(y)

        return self._dispatch_table[(op, device)](*args, **kwargs)

dispatcher = Dispatcher()