import numpy as np
import ctypes
from ars import *

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



testlib = ctypes.cdll.LoadLibrary('./ars.so')
class Data(ctypes.Structure): 
       _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                   ("hx", ctypes.POINTER(ctypes.c_double)),
                   ("hpx", ctypes.POINTER(ctypes.c_double))]
       

data = Data(np.ctypeslib.as_ctypes(x),
            np.ctypeslib.as_ctypes(hx),
            np.ctypeslib.as_ctypes(hpx))

class Bounds(ctypes.Structure): 
       _fields_ = [("lb", ctypes.c_bool),
                   ("xlb", ctypes.c_double),                   
                   ("ub", ctypes.c_bool),
                   ("xub", ctypes.c_double),
                   ("ifault", ctypes.c_int)]

b =  Bounds(lb=False,ub=False,ifault=0)
iwv=np.zeros(200,int)
rwv=np.zeros(700,float)
initial( ns, m, emax, data.x, data.hx, data.hpx, b.lb, b.xlb, b.ub, b.xub, b.ifault, iwv, rwv)

def h(x):
    yu=-x*x*0.5 
    return yu

def hprima(x)
    ypu=-x
    return ypu

num=200
sp=np.empty(num, dtype=float)
for i in range(num):
       sample(iwv,rwv,h,hprima,sim,ifault)
       sp[i]= sim
