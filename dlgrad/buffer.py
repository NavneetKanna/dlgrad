from __future__ import annotations

import atexit
import ctypes
import os


class Buffer:
    def __init__(self, data, temp_file_loc: str = "") -> None:
        self.buffer = data

        if temp_file_loc:
            self._temp_file_loc = temp_file_loc
            atexit.register(self._cleanup)

    def _cleanup(self):
        if os.path.exists(self._temp_file_loc):
            os.remove(self._temp_file_loc)

    @staticmethod
    def free(data) -> None:
        libc = ctypes.CDLL(None)  # Load the standard C library

        libc.free(data)
        data = None
