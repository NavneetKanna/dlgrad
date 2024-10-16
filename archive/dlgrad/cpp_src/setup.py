from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "interface",
        sorted(glob("*.cpp")),  # Sort source files for reproducibility
    ),
]

setup(
    name="dlgrad",
    ext_modules=ext_modules,
    python_requires="==3.10.14",
)
