import numpy as np

from ars import ARS

m=3
ns=100
emax=64
x=np.zeros(10, float)
hx=np.zeros(10, float)
hpx=np.zeros(10, float)
x[0]=0
x[1]=1.0
x[2]=-1.0
rwv=np.zeros(700, float)
iwv=np.zeros(200, np.int64)
def normal(x):
    return -x*x*0.5,-x

hx[0],hpx[0]=normal(x[0])
hx[1],hpx[1]=normal(x[1])
hx[2],hpx[2]=normal(x[2])
ars=ARS(ns,m,emax,x,hx,hpx,iwv,rwv)
print(ars.get_iwv())
print(ars.ifault)


import ctypes
testlib = ctypes.cdll.LoadLibrary('./st.so')
class Data(ctypes.Structure): 
       _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                   ("hx", ctypes.POINTER(ctypes.c_double)),
                   ("hpx", ctypes.POINTER(ctypes.c_double))]
       
data = Data(np.ctypeslib.as_ctypes(x),
            np.ctypeslib.as_ctypes(hx),
            np.ctypeslib.as_ctypes(hpx))


class Data(ctypes.Structure): 
       _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                   ("hx", ctypes.POINTER(ctypes.c_double)),
                   ("hpx", ctypes.POINTER(ctypes.c_double))]