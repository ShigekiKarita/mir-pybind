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
