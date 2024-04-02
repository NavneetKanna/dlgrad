"""
This should contain all buffer related tasks.

"""
from buffer.c_code import C
import subprocess

class Buffer:
    def __init__(self) -> None:
        pass

    def create_random_buffer(self, length: int):
        """
        subprocess.check_output(args=['clang', '-x', 'c', '-', '-shared', '-o', 'test1.so'], input=src.encode('utf-8'))

        """
        prg = C._random_buffer(length)