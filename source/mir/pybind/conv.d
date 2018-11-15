module mir.pybind.conv;

import std.string : toStringz;
import std.typecons : isTuple;
import std.conv : to;

import mir.ndslice : isSlice, SliceKind, Slice, Structure;
import mir.ndslice.connect.cpython : toPythonBuffer, fromPythonBuffer, PythonBufferErrorCode, PyBuf_indirect, PyBuf_format, PyBuf_writable, bufferinfo;
import deimos.python.Python;

import mir.pybind.format : formatTypes;

/// PythonBasicType is a type that can be converted into int/float/bool/str
enum bool isPythonBasicType(T) = !(isTuple!T || formatTypes!T == "O");

/// D type to PyObject conversion
auto toPyObject(double x) { return PyFloat_FromDouble(x); }

/// ditto
auto toPyObject(long x) { return PyLong_FromLongLong(x); }

/// ditto
auto toPyObject(ulong x) { return PyLong_FromUnsignedLongLong(x); }

/// ditto
static if (!is(ulong == size_t))
    auto toPyObject(size_t x) { return PyLong_FromSize_t(x); }

/// ditto
static if (!is(long == ptrdiff_t))
    auto toPyObject(ptrdiff_t x) { return PyLong_FromSsize_t(x); }

/// ditto
auto toPyObject(bool b) { return PyBool_FromLong(b ? 1 : 0); }

// FIXME: remove this line (need more version specification?)
extern (C) PyObject* PyUnicode_FromStringAndSize(const(char*) u, Py_ssize_t len);

/// ditto
auto toPyObject(string s) { return PyUnicode_FromStringAndSize(s.ptr, s.length); }

/// ditto
auto toPyObject(PyObject* p) { return p; }

/**
   default PyObject_GetBuffer flag
   TODO supportuser-defined flag
 */
enum PyBuf_full = PyBuf_indirect | PyBuf_format | PyBuf_writable;

/// numpy.ndarray.descr.type_num
enum NpyType : int
{
    npy_bool=0,
    npy_byte, npy_ubyte,
    npy_short, npy_ushort,
    npy_int, npy_uint,
    npy_long, npy_ulong,
    npy_longlong, npy_ulonglong,
    npy_float, npy_double, npy_longdouble,
    npy_cfloat, npy_cdouble, npy_clongdouble,
    npy_object=17,
    npy_string, npy_unicode,
    npy_void,
    /**
     * new 1.6 types appended, may be integrated
     * into the above in 2.0.
     */
    npy_datetime, npy_timedelta, npy_half,

    npy_ntypes,
    npy_notype,
    /// npy_char npy_attr_deprecate("use npy_string"),
    npy_userdef=256,  /* leave room for characters */

    /** the number of types not including the new 1.6 types */
    npy_ntypes_abi_compatible=21
};

/// e.g., bool -> npy_bool
template toNpyType(T)
{
    mixin("enum toNpyType = NpyType.npy_" ~ T.stringof ~ ";");
}

/// mir.ndslice.Slice to PyObject conversion
auto toPyObject(T, size_t n, SliceKind k)(Slice!(T*, n, k) x) if (isPythonBasicType!T)
{
    bufferinfo buf;
    Structure!n str;
    auto err = toPythonBuffer(x, buf, PyBuf_full, str);
    if (err != PythonBufferErrorCode.success) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unable to convert Slice object into Py_buffer");
    }
    return PyMemoryView_FromBuffer(cast(Py_buffer*) &buf);
    // FIXME use Array API
    // auto p = PyArray_SimpleNew(n, cast(npy_intp*) x.shape.ptr, toNpyType!T);
}

/// std.typecons.Tuple to PyObject conversion
auto toPyObject(T)(T xs) if (isTuple!T)
{
    enum N = T.length;
    auto p = PyTuple_New(N);
    if (p == null) {
        PyErr_SetString(PyExc_RuntimeError,
                        ("unable to allocate " ~ N.to!string ~ " tuple elements").toStringz);
    }
    static foreach (i; 0 .. T.length) {{
        auto pi = toPyObject(xs[i]);
        PyTuple_SetItem(p, i, pi);
     }}
    return p;
}

template PointerOf(T)
{
    import mir.pybind.format : formatTypes;
    static if (isPythonBasicType!T)
        alias PointerOf = T*;
    else
        alias PointerOf = PyObject**;
}

/// PyObject to D object conversion for python basic types
auto fromPyObject(T)(ref T x, PyObject* o) if (isPythonBasicType!T) { return ""; }

/// PyObject to D object conversion for mir slice
auto fromPyObject(T)(ref T x, PyObject* o) if (isSlice!T)
{
    bufferinfo buf;
    if (PyObject_CheckReadBuffer(o) == -1) {
        return "unable to read buffer";
    }
    PyObject_GetBuffer(o, cast(Py_buffer*) &buf, PyBuf_full);
    scope(exit) PyBuffer_Release(cast(Py_buffer*) &buf);
    auto err = fromPythonBuffer(x, buf);
    if (err != PythonBufferErrorCode.success)
    {
        return "fail to execute mir.ndslice.connect.cpython.fromPythonBuffer (PythonBufferErrorCode: "
            ~ err.to!string ~ ")";
    }
    return "";
}

/**
   D function to PyObject conversion
 */
extern(C)
PyObject* toPyFunction(alias dFunction)(PyObject* mod, PyObject* args)
{
    import std.conv : to;
    import std.traits : Parameters, ReturnType;
    import std.meta : staticMap;
    import std.typecons : Tuple;
    import std.string : toStringz, replace;
    import mir.pybind.format : formatTypes;

    alias Ps = Parameters!dFunction;
    Tuple!Ps params;
    alias Ptrs = staticMap!(PointerOf, Ps);
    Tuple!Ptrs ptrs;

    static foreach (i; 0 .. Ps.length)
    {
        static if (isPythonBasicType!(Ps[i]))
        {
            ptrs[i] = &params[i];
        }
        else
        {
            mixin(
                q{
                    PyObject* obj$;
                    ptrs[i] = &obj$;
                }.replace("$", i.to!string));

        }
    }
    if (!PyArg_ParseTuple(args, formatTypes!(Ps).toStringz, ptrs.expand))
    {
        return null;
    }
    else
    {
        static foreach (i; 0 .. Ps.length)
        {
            static if (!isPythonBasicType!(Ps[i])) {{
                mixin(
                    q{
                        auto msg = params[i].fromPyObject(obj$);
                        if (msg != "")
                        {
                            auto e = "incompatible array object, expected type: "
                                ~ Ps[i].stringof ~ ", message: " ~ msg;
                            PyErr_SetString(PyExc_RuntimeError, e.toStringz);
                        }
                    }.replace("$", i.to!string));
                }}
        }

        alias R = ReturnType!dFunction;
        static if (is(R == void))
        {
            dFunction(params.expand);
            return newNone(); // TODO support return values
        }
        else
        {
            return toPyObject(dFunction(params.expand));
        }
    }
    assert(false);
}
