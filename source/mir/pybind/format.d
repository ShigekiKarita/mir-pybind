module mir.pybind.format;

import std.traits;
import std.typecons;
import mir.ndslice : isSlice;

template formatTypes(Ts...) {
    enum formatTypes = {
        string ret;
        size_t cnt;
        static foreach (T; Ts) {
            static if (is(T == string)) {
                ret ~= "s";
                ++cnt;
            }
            else static if (isBoolean!T) {
                ret ~= "p";
                ++cnt;
            }
            else static if (isIntegral!T) {
                ret ~= "L"; // long long
                ++cnt;
            }
            else static if (isFloatingPoint!T) {
                ret ~= "d";
                ++cnt;
            }
            else static if (isTuple!T) {
                ret ~= "(";
                ret ~= formatTypes!(T.Types).str;
                ret ~= ")";
            }
            else {
                ret ~= "O";
                ++cnt;
                // static assert(false, "unknown type to format: " ~ T.stringof);
            }
        }
        return tuple!("str", "count")(ret, cnt);
    }();
}

unittest {
    static assert(formatTypes!(string, bool, double, int, PyObject*).str == "spdLO");
    static assert(formatTypes!(Tuple!(bool, double).Types).str == "pd");
    static assert(formatTypes!(string, Tuple!(bool, Tuple!(double, long)).str, int, PyObject*)
                  == "s(p(dL))LO");
}
