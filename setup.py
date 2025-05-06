import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build import build as _build


class BuildWithMetal(_build):
    def run(self):  # noqa: ANN201
        _build.run(self)

        package_dir = os.path.join(self.build_lib, 'dlgrad')
        metal_dir = os.path.join(package_dir, 'src', 'metal')
        os.makedirs(metal_dir, exist_ok=True)

        metal_file = os.path.join('dlgrad', 'src', 'metal', 'arithmetic.metal')
        ir_file = os.path.join(package_dir, 'arithmetic.ir')
        metallib_file = os.path.join(metal_dir, 'arithmetic.metallib')
        subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c',  metal_file])
        subprocess.check_call(['xcrun',  '-sdk', 'macosx', 'metallib','-o', metallib_file, ir_file])
        os.remove(ir_file)

        metal_file = os.path.join('dlgrad', 'src', 'metal', 'utils.metal')
        ir_file = os.path.join(package_dir, 'utils.ir')
        metallib_file = os.path.join(metal_dir, 'utils.metallib')
        subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c',  metal_file])
        subprocess.check_call(['xcrun',  '-sdk', 'macosx', 'metallib','-o', metallib_file, ir_file])
        os.remove(ir_file)

        metal_file = os.path.join('dlgrad', 'src', 'metal', 'sum.metal')
        ir_file = os.path.join(package_dir, 'sum.ir')
        metallib_file = os.path.join(metal_dir, 'sum.metallib')
        subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c',  metal_file])
        subprocess.check_call(['xcrun',  '-sdk', 'macosx', 'metallib','-o', metallib_file, ir_file])
        os.remove(ir_file)

        metal_file = os.path.join('dlgrad', 'src', 'metal', 'max.metal')
        ir_file = os.path.join(package_dir, 'max.ir')
        metallib_file = os.path.join(metal_dir, 'max.metallib')
        subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c',  metal_file])
        subprocess.check_call(['xcrun',  '-sdk', 'macosx',  'metallib','-o', metallib_file, ir_file])
        os.remove(ir_file)

        metal_file = os.path.join('dlgrad', 'src', 'metal', 'matmul.metal')
        ir_file = os.path.join(package_dir, 'matmul.ir')
        metallib_file = os.path.join(metal_dir, 'matmul.metallib')
        subprocess.check_call(['xcrun', '-sdk', 'macosx', 'metal', '-O2', '-o', ir_file, '-c',  metal_file])
        subprocess.check_call(['xcrun',  '-sdk', 'macosx',  'metallib','-o', metallib_file, ir_file])
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
