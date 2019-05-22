import std.stdio;
import mir.ndslice;
import mir.pybind : def, defModule;
import deimos.python.Python; //  : PyMethodDef, METH_VARARGS;

// https://docs.python.jp/3/extending/newtypes_tutorial.html
struct Custom {
    int number;

    ref add(int n) {
        number += n;
        return this;
    }
}


// TODO auto generate these wrappers of Custom
struct PyTypeWrapper(T) {
    struct PyStruct {
        mixin PyObject_HEAD;
        T dtype;
        alias dtype this;
    }

    PyTypeObject pytype() {
        PyTypeObject pytype;
        Py_SET_TYPE(&pytype, &PyType_Type);
        pytype.tp_basicsize = PyStruct.sizeof;
        pytype.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        pytype.tp_methods = _methods.ptr;
        pytype.tp_name = "rawexample.Derived";
        pytype.tp_new = &PyType_GenericNew;
        // TODO
        // pytype.tp_base = &Base_type;
        PyType_Ready(&pytype);
        Py_INCREF(cast(PyObject*)&pytype);
        // PyModule_AddObject(m, "Derived", cast(PyObject*)&pytype);
        return pytype;
    }

    static PyMemberDef[] _members = [];

    extern (C)
    static PyObject* _add(PyObject* self, PyObject* args) {
        return self;
    }

    static PyMethodDef[] _methods = [
        {"name", &_add, METH_VARARGS, "add int to CustomObject.number"},
        {null} // sentinel
        ];
}

alias PyCustom = PyTypeWrapper!Custom;
// static PyTypeCustom = pyTypeOf!Custom();


double foo(long x) {
    return x * 2;
}

string baz(double d, char[] s) {
    import std.conv : to;
    return d.to!string  ~ s.to!string;
}

import std.typecons;
auto bar(long i, double d, Tuple!(bool, Slice!(double*, 1)) t) {
    writeln(t[1]);
    return tuple(i, tuple(tuple(d, i, t[0], t[1][0])));
}

// NOTE: returning slice is partially supported (need numpy.asarray in python)
Slice!(double*, 2) sum(Slice!(double*, 2) x, Slice!(double*, 1) y) {
    auto z = x.slice; // copy
    foreach (zi; z) {
        zi[] += y;
    }
    return z;
}

mixin defModule!(
    "libtest_module", // module name
    "this is D module", // module doc
    // register d-func and doc under the module
    [def!(foo, typeof(foo).stringof),
     def!(baz, "this is baz"),
     def!(bar, "this is bar"),
     def!(sum, "this is sum")]);

static this() {
    writeln(typeof(foo).stringof);
}

/* this mixin generates following for example

extern (C):
static PyModuleDef mod = {
    PyModuleDef_HEAD_INIT,
    "mod",                      // module name
    "this is D language mod",   // module doc
    -1,                         // size of per-interpreter state of the module,
                                // or -1 if the module keeps state in global variables.
};

static methods = [
    def!(foo, "this is foo"),
    ...
    PyMethodDef_SENTINEL
    ];

auto PyInit_libtest_module() {
    import core.runtime : rt_init;
    rt_init();
    Py_AtExit(&rtAtExit);
    mod.m_methods = methods.ptr;
    return PyModule_Create(&mod);
}

*/
