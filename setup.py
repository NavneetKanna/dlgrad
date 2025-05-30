import os
import platform
import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build import build as _build

if platform.system() == 'Windows':
    print("dlgrad does not support Windows. Please use a Unix-based system (macOS/Linux).")
    sys.exit(1)


METAL_FILES = [
    'arithmetic', 'utils', 'sum', 'max', 'matmul', 'transpose'
]

def metal_tool_available(tool):  # noqa: ANN001, ANN201
    return shutil.which(tool) is not None

class BuildWithMetal(_build):
    def run(self):  # noqa: ANN201
        _build.run(self)

        if platform.system() != 'Darwin':
            return

        if not (metal_tool_available('xcrun') and metal_tool_available('metal')):
            print("xcrun/metal not found — skipping Metal backend build.")
            return

        package_dir = os.path.join(self.build_lib, 'dlgrad')
        metal_dir = os.path.join(package_dir, 'src', 'metal')
        os.makedirs(metal_dir, exist_ok=True)

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        for name in ['arithmetic', 'utils', 'sum', 'max', 'matmul', 'transpose']:
            metal_file = os.path.join('dlgrad', 'src', 'metal', f'{name}.metal')
            ir_file = os.path.join(build_temp, f'{name}.ir')
            metallib_file = os.path.join(metal_dir, f'{name}.metallib')

            subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c', metal_file])
            subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metallib', '-o', metallib_file, ir_file])
            os.remove(ir_file)

setup(
    name='dlgrad',
    description='An autograd engine built for my understanding',
    author='Navneet Kanna',
    license='MIT',
    version='0.5',
    packages=find_packages(),
    python_requires='>=3.10',
    cffi_modules=[
        'dlgrad/builder/float_rand_build.py:ffi',
        'dlgrad/builder/float_arithmetic_build.py:ffi',
        'dlgrad/builder/float_utils_build.py:ffi',
        'dlgrad/builder/float_matmul_build.py:ffi',
        'dlgrad/builder/float_sum_build.py:ffi',
        'dlgrad/builder/float_max_build.py:ffi',
        'dlgrad/builder/float_full_build.py:ffi',
        "dlgrad/builder/float_activation_functions_build.py:ffi",
        'dlgrad/builder/float_comparision_build.py:ffi',
        'dlgrad/builder/float_allocate_build.py:ffi',
        'dlgrad/builder/float_loss_build.py:ffi',
        'dlgrad/builder/float_mnist_build.py:ffi',
        'dlgrad/builder/float_transpose_build.py:ffi',
    ],
    setup_requires=['cffi>=1.17.1'],
    install_requires=['cffi>=1.0.0', 'requests', 'pyobjc-framework-Metal'],
    cmdclass={'build': BuildWithMetal},
)
