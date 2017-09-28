import numpy as np
from ars import *
import ctypes
m = 3
ns = 100
emax = 64.

x = np.zeros(10, float)
hx = np.zeros(10, float)
hpx = np.zeros(10, float)

x[0] = 0.
x[1] = 1.0
x[2] = -1.0
def pynormal(x):
    return -x*x*0.5,-x

hx[0], hpx[0] = pynormal(x[0])
hx[1], hpx[1] = pynormal(x[1])
hx[2], hpx[2] = pynormal(x[2])
print x
print hx
print hpx

num = 20
sp=py_ars(ns, m, emax, x, hx, hpx, num, normal_ctypes )

print(sp)
