import platform
import sys

from setuptools import find_packages, setup

if platform.system() == 'Windows':
    print("dlgrad does not support Windows. Please use a Unix-based system (macOS/Linux).")
    sys.exit(1)

setup(
    name='dlgrad',
    description='An autograd engine built for my understanding',
    author='Navneet Kanna',
    license='MIT',
    version='0.9',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['cffi>=1.0.0', 'tqdm', 'pyobjc-framework-Metal; sys_platform == "darwin"'],
)
