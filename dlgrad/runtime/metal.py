# https://dougallj.github.io/applegpu/docs.html

import sysconfig

import Metal

metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/add.metallib"

device = Metal.MTLCreateSystemDefaultDevice()
commandQueue = device.newCommandQueue()  # noqa: N816
lib = device.newLibraryWithURL_error_(metallib_path, None)[0]
func_name = lib.newFunctionWithName_("add_arrays")
print(lib)
print(lib._.functionNames)

