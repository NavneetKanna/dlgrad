import subprocess
from dlgrad.helpers import BroadcastHelper, check_temp_file_exists, get_temp_loc
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
import tempfile
import os
import ctypes

def compile_add_axis0():
    pass
    # prg = C.add_axis0(c_dtype, out_len=BroadcastHelper.out_len)
    # name = f"cpu_{c_dtype}_add0"
    # temp_file = check_temp_file_exists(starts_with=name)
