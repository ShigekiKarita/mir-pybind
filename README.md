# mir-pybind
[![Build Status](https://travis-ci.org/ShigekiKarita/mir-pybind.svg?branch=master)](https://travis-ci.org/ShigekiKarita/mir-pybind)
[![dub](https://img.shields.io/dub/v/mir-pybind.svg)](https://code.dlang.org/packages/mir-pybind)


Like pybind11 in C++, mir-pybind provides simplest communication interface between D-language (e.g., mir.ndslice) and python (e.g., numpy, PIL) by [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol) and PyObject based conversion. Unlike my previous project [mir-pybuffer](https://github.com/ShigekiKarita/mir-pybuffer), this does not need python library because this is pure D project.

## usage

### python side

All you need is importing D dynamic library. Wrong arguments will raise TypeError, RuntimeError, etc from D library.

``` python
import numpy
import libtest_module

assert libtest_module.__doc__ == "this is D module"
assert libtest_module.foo(1) == 2.0
assert libtest_module.baz(1.5, "MB") == "1.5MB"

x = numpy.array([[0, 1, 2], [3, 4, 5]]).astype(numpy.float64)
y = numpy.array([0, 1, 2]).astype(numpy.float64)
mem = numpy.asarray(libtest_module.sum(x, y))
numpy.testing.assert_allclose(mem, x + y)

assert libtest_module.bar(1, 2.0, (True, y))  == (1, ((2.0, 1, True, 0.0),))

print("TEST OK")
```

### D side

You need to implement funcitons and register them by `mir.pybind.defModule`.

``` d
import std.stdio;
import mir.ndslice;
import mir.pybind : def, defModule;

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
```

To create dynamic library, you also need to add `"targetType": "dynamicLibrary"` in [dub.json](test/dub.json).

## detail

The mixin in the last line generates following for example.

``` d
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
```


## roadmap

- [DONE] support basic type argument and return: int, float, bool, str, tuple
- [DONE] support numpy/ndslice argument
- [DONE] support numpy/ndslice return (NOTE: need numpy.asarray for returned values)
- [TODO] user-defined class/struct support
