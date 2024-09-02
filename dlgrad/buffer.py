from __future__ import annotations

import atexit
import ctypes
import os
from itertools import chain


class Buffer:
    def __init__(self, data, temp_file_loc: str = "") -> None:
        self.buffer = data

        if temp_file_loc:
            self._temp_file_loc = temp_file_loc
            atexit.register(self._cleanup)

    def _cleanup(self):
        if os.path.exists(self._temp_file_loc):
            os.remove(self._temp_file_loc)

    def create_buf_from_list(x: list) -> Buffer:
        if not len(set(map(len, x))) == 1:
            # TODO: raise error
            print("all len should be equal")

        x = [*chain(*x)] # flatten into 1d

    def create_buf_from_idx(x: "Tensor", y: "Tensor") -> Buffer: # noqa: F821 # type: ignore
        pass

    @staticmethod
    def free(data) -> None:
        libc = ctypes.CDLL(None)  # Load the standard C library

        libc.free(data)
        data = None
