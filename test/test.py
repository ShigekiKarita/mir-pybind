import numpy
import libtest_module

assert libtest_module.__doc__ == "this is D module"
assert libtest_module.foo(1) == 2.0

# print(libtest_module.baz(1.5, "MB"))
#assert libtest_module.baz(1.5, "MB") == "1.5MB"

x = numpy.array([[0, 1, 2], [3, 4, 5]]).astype(numpy.float64)
y = numpy.array([0, 1, 2]).astype(numpy.float64)
mem = libtest_module.sum(x, y)
numpy.testing.assert_allclose(numpy.frombuffer(mem, dtype=numpy.float64), [3, 6, 9])

print(libtest_module.bar.__doc__)
assert libtest_module.bar(1, 2.0, (True, 4))  == (1, ((2.0, 1, (True, 4)),))

print("TEST OK")
