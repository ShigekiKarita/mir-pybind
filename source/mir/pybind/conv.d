module mir.pybind.conv;

import std.stdio;
import std.string : toStringz;
import std.traits : staticMap;
import std.typecons : isTuple, Tuple;
import std.conv : to;
import std.format : format;

import mir.ndslice : isSlice, SliceKind, Slice, Structure;
import mir.ndslice.connect.cpython : toPythonBuffer, fromPythonBuffer, PythonBufferErrorCode, PyBuf_indirect, PyBuf_format, PyBuf_writable, bufferinfo;

import deimos.python.Python;
// FIXME: remove this line (need more version specification?)
extern (C) PyObject* PyUnicode_FromStringAndSize(const(char*) u, Py_ssize_t len);
extern (C) const(char*) PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size);
extern (C) const(char*) PyUnicode_AsUTF8(PyObject *unicode);

import mir.pybind.format : formatTypes;


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
auto toPyObject(T, size_t n, SliceKind k)(Slice!(T*, n, k) x)
{
    // import mir.ndslice.connect.cpython : cannot_create_f_contiguous_buffer;
    bufferinfo buf;
    Structure!n str;
    auto err = toPythonBuffer(x, buf, PyBuf_full, str);
    if (err != PythonBufferErrorCode.success) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unable to convert Slice object into Py_buffer");
    }
    return PyMemoryView_FromBuffer(cast(deimos.python.object.Py_buffer*) &buf);
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

/// PyObject to D object conversion for python basic types
auto fromPyObject(ref long x, PyObject* o)
{
    x = PyLong_AsLongLong(o);
    return "";
}

/// ditto
auto fromPyObject(ref ulong x, PyObject* o)
{
    x = PyLong_AsUnsignedLongLong(o);
    return "";
}

/// ditto
auto fromPyObject(ref double x, PyObject* o)
{
    x = PyFloat_AsDouble(o);
    return "";
}

/// ditto
auto fromPyObject(ref bool x, PyObject* o)
{
    x = PyObject_IsTrue(o) == 1;
    return "";
}

/// ditto
auto fromPyObject(ref char[] x, PyObject* o)
{
    import std.string : fromStringz;
    // TODO find better way
    // FIXME need incref here?
    Py_ssize_t len;
    auto ptr = cast(char*) PyUnicode_AsUTF8AndSize(o, &len);
    if (ptr == null) return "invalid UTF8 string";
    x = ptr[0 .. len];
    return "";
}

/// PyObject to D object conversion for mir slice
auto fromPyObject(T)(ref T x, PyObject* o) if (isSlice!T)
{
    bufferinfo buf;
    if (PyObject_CheckReadBuffer(o) == -1)
    {
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

/// PyObject to D object conversion for std.typecons.tuple
auto fromPyObject(T)(ref T xs, PyObject* os) if (isTuple!T)
{
    import deimos.python.tupleobject;
    if (!PyTuple_Check(os))
    {
        return "non-tuple python object is passed to std.typecons.Tuple argument";
    }
    if (PyTuple_Size(os) != xs.length)
    {
        return "lengths of tuples are not matched: python %s vs D %s".format(
            PyTuple_Size(os), xs.length);
    }
    foreach (i, ref x; xs)
    {
        auto o = PyTuple_GetItem(os, i);
        x.fromPyObject(Py_XINCREF(o));
    }
    return "";
}

/**
   D function to PyObject conversion
 */
extern(C)
PyObject* toPyFunction(alias dFunction)(PyObject* mod, PyObject* args)
{
    import std.stdio;
    import std.conv : to;
    import std.traits : Parameters, ReturnType;
    import std.meta : staticMap;
    import std.typecons : Tuple, tuple;
    import std.string : toStringz, replace;
    import mir.pybind.format : formatTypes;

    alias Ps = Parameters!dFunction;
    Tuple!Ps params;

    PyObject*[Ps.length] pyobjs;
    mixin(
        {
            string fmt = "auto fmt = \"";
            string tup = "auto pyrefs = tuple(";
            foreach (i; 0 .. Ps.length) {
                fmt ~= "O";
                tup ~= "&pyobjs[" ~ i.to!string ~ "],";
            }
            return tup[0 .. $-1] ~ ");" ~ fmt ~ "\".toStringz;";
        }());

    // TODO keyword args

    // TODO typecheck error message (maybe pyrefs.length is shorter?)
    // enum f = formatTypes!Ps;
    // void*[f.count] _refs;
    // if (!PyArg_ParseTuple(args, f.str.toStringz, null))
    // {
    //     return null; // parsing failure
    // }

    // extract PyObject* s from args
    if (!PyArg_ParseTuple(args, fmt, pyrefs.expand))
    {
        return null; // parsing failure
    }

    static foreach (i; 0 .. Ps.length)
    {
        {
            const msg = params[i].fromPyObject(pyobjs[i]);
            if (msg != "")
            {
                auto e = "incompatible object, expected type: "
                    ~ Ps[i].stringof ~ ", message: " ~ msg;
                PyErr_SetString(PyExc_RuntimeError, e.toStringz);
            }
        }
    }

    static if (is(ReturnType!dFunction == void))
    {
        dFunction(params.expand);
        return Py_None();
    }
    else
    {
        return toPyObject(dFunction(params.expand));
    }
}
