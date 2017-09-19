import numpy as np
from ars import main

m = 3
ns = 100
emax = 64

x = np.zeros(10, float)
hx = np.zeros(10, float)
hpx = np.zeros(10, float)

x[1] = 0
x[2] = 1.0
x[3] = -1.0
print x
def normal(x):
    return -x*x*0.5,-x

hx[1], hpx[1] = normal(x[1])
hx[2], hpx[2] = normal(x[2])
hx[3], hpx[3] = normal(x[3])

def h(x):
    yu = -x*x*0.5
    return yu

def hprima(x):
    ypu = -x
    return ypu

num = 20
sp = main(ns, m, emax, x, hx, hpx, num, h, hprima)
print(sp)
