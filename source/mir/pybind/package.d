module mir.pybind;
/** CODING GUIDE

D or Python:

When we can implement something both in D or Python, we do everything in D.
D has less overhead than Python (the reason why we use D from Python).

Error Handling:

D function should raise exception for python by PyErr_SetString(PyObject* type, const char* message)
The type can be PyExc_RuntimeError, PyExc_TypeError, etc
see also https://docs.python.org/3/c-api/exceptions.html#standard-exceptions
 */

private import deimos.python.Python : PyMethodDef, METH_VARARGS;
private import mir.pybind.conv : toPyFunction;

/// template to define python function (wrapper of PyMethodDef)
/// TODO insert type signature and args in docstring automatically
enum def(alias dfunc, string doc = "", string name = __traits(identifier, dfunc))
    = PyMethodDef(name, &toPyFunction!dfunc, METH_VARARGS, doc);

/// template to define python module (wrapper of PyModuleDef)
mixin template defModule(string modName, string modDoc, PyMethodDef[] defs)
{
    private import deimos.python.Python : PyModuleDef, PyModuleDef_Base, PyMethodDef;
    private import std.string : replace;

    extern (C):
    enum PyModuleDef_Base PyModuleDef_HEAD_INIT = {{1, null}, null, 0, null};

    enum PyMethodDef PyMethodDef_SENTINEL = {null, null, 0, null};

    void rtAtExit()
    {
        import core.runtime : rt_term;
        rt_term();
    }

    static PyModuleDef mod = {PyModuleDef_HEAD_INIT, m_name: modName, m_doc: modDoc, m_size: -1};
    static methods = defs ~ [PyMethodDef_SENTINEL];

    mixin(
        q{
            pragma(mangle, __traits(identifier, PyInit_$))
            auto PyInit_$()
            {
                import deimos.python.Python : Py_AtExit, PyModule_Create;
                import core.runtime : rt_init;
                rt_init();
                Py_AtExit(&rtAtExit);
                mod.m_methods = methods.ptr;
                // TODO import_array(); // enable numpy API
                return PyModule_Create(&mod);
            }
        }.replace("$", modName));
}
