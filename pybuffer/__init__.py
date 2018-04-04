"""
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import ctypes


# see also
# https://docs.python.org/3/c-api/buffer.html#buffer-request-types
# $CONDA_PREFIX/include/python3.6m/object.h
PyBUF_SIMPLE   = 0
PyBUF_WRITABLE = 0x0001
PyBUF_FORMAT   = 0x0004
PyBUF_ND       = 0x0008
PyBUF_STRIDES  = 0x0010 | PyBUF_ND

PyBUF_C_CONTIGUOUS   = 0x0020 | PyBUF_STRIDES
PyBUF_F_CONTIGUOUS   = 0x0040 | PyBUF_STRIDES
PyBUF_ANY_CONTIGUOUS = 0x0080 | PyBUF_STRIDES
PyBUF_INDIRECT       = 0x0100 | PyBUF_STRIDES

PyBUF_CONTIG_RO  = PyBUF_ND
PyBUF_CONTIG     = PyBUF_ND | PyBUF_WRITABLE

PyBUF_STRIDED_RO = PyBUF_STRIDES
PyBUF_STRIDED    = PyBUF_STRIDES | PyBUF_WRITABLE

PyBUF_RECORDS_RO = PyBUF_STRIDES | PyBUF_FORMAT
PyBUF_RECORDS    = PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE

PyBUF_FULL_RO = PyBUF_INDIRECT | PyBUF_FORMAT
PyBUF_FULL    = PyBUF_INDIRECT | PyBUF_FORMAT | PyBUF_WRITABLE

Py_ssize_t = ctypes.c_ssize_t
Py_ssize_t_p = ctypes.POINTER(Py_ssize_t)


class PyBuffer(ctypes.Structure):
    """Python Buffer Interface
    See_also:
    https://docs.python.org/3/c-api/buffer.html#buffer-protocol
    $CONDA_PREFIX/include/python3.6m/object.h
    """
    _fields_ = (('buf', ctypes.c_void_p),
                ('obj', ctypes.c_void_p), # owned reference
                ('len', Py_ssize_t),
                ('itemsize', Py_ssize_t),

                ('readonly', ctypes.c_int),
                ('ndim', ctypes.c_int),
                ('format', ctypes.c_char_p),
                ('shape', Py_ssize_t_p),
                ('strides', Py_ssize_t_p),
                ('suboffsets', Py_ssize_t_p),
                ('internal', ctypes.c_void_p))

    def __init__(self, obj, flags=PyBUF_FULL):
        pyapi.PyObject_GetBuffer(obj, ctypes.byref(self), flags)

    def __del__(self):
        pyapi.PyBuffer_Release(ctypes.byref(self))
        ctypes.memset(ctypes.byref(self), 0, ctypes.sizeof(self))


def to_buffer(obj, flags=PyBUF_FULL):
    return ctypes.byref(PyBuffer(obj, flags))


def check_buffer(obj):
    try:
        memoryview(obj)
        return True
    except TypeError:
        return False


class CDLL(ctypes.CDLL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize D runtime
        if hasattr(self, "rt_init"):
            assert self.rt_init()

    def __del__(self):
        if hasattr(self, "rt_term"):
            self.rt_term()

    # https://github.com/python/cpython/blob/306559e6ca15b86eb230609f484f48132b7ca383/Lib/ctypes/__init__.py#L311
    def __getattr__(self, name):
        newname = "pybuffer_" + name
        try:
            func = super().__getattr__(newname)
        except AttributeError:
            func = super().__getattr__(name)

        def wrapped(*args):
            newargs = []
            for a in args:
                # see https://docs.python.org/3/c-api/buffer.html#c.PyObject_CheckBuffer
                if check_buffer(a) == 1:
                    newargs.append(to_buffer(a))
                else:
                    newargs.append(a)
            return func(*newargs)
        setattr(self, name, wrapped)
        return wrapped


# PyBuffer functions in PythonAPI
pyapi = ctypes.PyDLL("PythonAPI", handle=ctypes.pythonapi._handle)
pyapi.PyObject_GetBuffer.argtypes = (ctypes.py_object,          # obj
                                     ctypes.POINTER(PyBuffer),  # view
                                     ctypes.c_int)              # flags
pyapi.PyBuffer_Release.argtypes = ctypes.POINTER(PyBuffer),     # view
