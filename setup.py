from setuptools import find_packages, setup

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
        'dlgrad/builder/float_add_build.py:ffi', 
        'dlgrad/builder/float_arithmetic_build.py:ffi', 
        'dlgrad/builder/float_neg_build.py:ffi',
        'dlgrad/builder/float_matmul_build.py:ffi',
        'dlgrad/builder/float_sum_build.py:ffi',
        'dlgrad/builder/float_full_build.py:ffi',
    ],
    setup_requires=['cffi>=1.17.1'],
    install_requires=['cffi>=1.0.0']
)
