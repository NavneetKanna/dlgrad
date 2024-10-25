from setuptools import setup, find_packages

setup(
    name='dlgrad',
    description='An autograd engine built for my understanding',
    author='Navneet Kanna',
    license='MIT',
    version='0.5',
    packages=find_packages(),
    python_requires='>=3.10',
    cffi_modules=['builder/float_rand_build.py:ffi'],
    setup_requires=['cffi>=1.0.0'],
    install_requires=['cffi>=1.0.0']
)
