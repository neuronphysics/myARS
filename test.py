import numpy as np
from ars import main

m = 3
ns = 100
emax = 64

x = np.zeros(10, float)
hx = np.zeros(10, float)
hpx = np.zeros(10, float)

x[0] = 0
x[1] = 1.0
x[2] = -1.0

def normal(x):
    return -x*x*0.5,-x

hx[0], hpx[0] = normal(x[0])
hx[1], hpx[1] = normal(x[1])
hx[2], hpx[2] = normal(x[2])

def h(x):
    yu = -x*x*0.5
    return yu

def hprima(x):
    ypu = -x
    return ypu

num = 5
sp = main(ns, m, emax, x, hx, hpx, num, h, hprima)
print(sp)
