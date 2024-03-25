import ctypes as ct

# TODO: Complie so with 02 and test 
class c_rand_buffer:
    dll = ct.CDLL('./backend/c/rand_buffer.so')
    dll.create_rand_buffer.argtypes = (ct.c_int,)
    dll.create_rand_buffer.restype = ct.POINTER(ct.c_float)  

    @staticmethod
    def _create(size): return c_rand_buffer.dll.create_rand_buffer(size)

class c_add:
    dll = ct.CDLL('./backend/c/add.so')
    dll.add.argtypes = (ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int) # TODO: be explicit, int32 ?
    dll.add.restype = ct.POINTER(ct.c_float)  

    @staticmethod
    def _add(arr1, arr2, length): return c_add.dll.add(arr1, arr2, length)

class c_free:
    dll = ct.CDLL('./backend/c/free.so')
    dll.free_buf.argtypes = ct.c_void_p,
    dll.free_buf.restype = None

    @staticmethod
    def _free(buf_p): c_free.dll.free_buf(buf_p) 
