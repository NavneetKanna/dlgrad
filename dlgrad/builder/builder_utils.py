from pathlib import Path

from cffi import FFI

ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / "src" / "c"

def build_extension(module_name: str, headers: str, sources: str, cdef: str) -> FFI:
    ffi = FFI()
    ffi.cdef(cdef)

    ffi.set_source(
        f"dlgrad.{module_name}",
        "\n".join(f'#include "{SRC_DIR / header}"' for header in headers),
        sources=[str(SRC_DIR / src) for src in sources],
        extra_compile_args=["-O2", "-march=native"]
    )
    return ffi
