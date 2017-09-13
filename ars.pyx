#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
from cpython cimport array
import cython
import numpy as np
import ctypes
cimport numpy as np
cdef extern from "math.h":
     cpdef double log(double x)
     cpdef double exp(double x)

cdef extern from "stdlib.h":
     cpdef int rand()
     cpdef enum: RAND_MAX

from libc.math cimport fabs

cdef void initial(int ns, int m, double emax, double* x, double* hx, double*
        hpx, int lb, double xlb, int ub, double xub, int* ifault, double* iwv,
        double* rwv):
      """
      This subroutine takes as input the number of starting values m
      and the starting values x(i), hx(i), hpx(i)  i = 1, m
      As output we have pointer iipt along with ilow and ihigh and the lower
      and upper hulls defined  by z, hz, scum, cu, hulb, huub stored in working
      vectors iwv and rwv
      Ifault detects wrong starting points or non-concavity
      ifault codes, subroutine initial
      0:successful initialisation
      1:not enough starting points
      2:ns is less than m
      3:no abscissae to left of mode (if lb = false)
      4:no abscissae to right of mode (if ub = false)
      5:non-log-concavity detect
      """
      cdef int nn, ilow, ihigh, i
      cdef int iipt, iz, ihuz, iscum, ix, ihx, ihpx
      cdef bint horiz
      cdef double hulb, huub, eps, cu, alcu, huzmax
      
      """
      DESCRIPTION OF PARAMETERS and place of storage

      lb   iwv[4] : boolean indicating if there is a lower bound to the
                     domain
      ub   iwv[5] : boolean indicating if there is an upper bound
      xlb  rwv[7] : value of the lower bound
      xub  rwv[8] : value of the upper bound
      emax rwv[2] : large value for which it is possible to compute
                    an exponential, eps = exp(-emax) is taken as a small
                    value used to test for numerical unstability
      m    iwv[3] : number of starting points
      ns   iwv[2] : maximum number of points defining the hulls
      x    rwv(ix+1)  : vector containing the abscissae of the starting
                        points
      hx   rwv(ihx+1) : vector containing the ordinates
      hpx  rwv(ihpx+1): vector containing the derivatives
      ifault      : diagnostic
      iwv, rwv     : integer and real working vectors
      """
      eps = expon(-emax, emax)
      ifault[0] = 0
      ilow = 0
      ihigh = 0
      nn = ns+1
      #at least one starting point
      if (m < 1):
         ifault[0] = 1

      huzmax = hx[0]
      if not ub:
         xub = 0.0

      if not lb:
         xlb = 0.0

      hulb = (xlb-x[0])*hpx[0] + hx[0]
      huub = (xub-x[0])*hpx[0] + hx[0]
      #if bounded on both sides
      if (ub and lb):
         huzmax = max(huub, hulb)
         horiz = (fabs(hpx[0]) < eps)
         if (horiz):
           cu = expon((huub+hulb)*0.5-huzmax, emax)*(xub-xlb)
         else:
           cu = expon(huub-huzmax, emax)*(1-expon(hulb-huub, emax))/hpx[0]
      elif ((ub) and (not lb)):
         #if bounded on the right and unbounded on the left
         huzmax = huub
         cu = 1.0/hpx[0]

      elif ((not ub) and (lb)):
         #if bounded on the left and unbounded on the right
         huzmax = hulb
         cu = -1.0/hpx[0]

         #if unbounded at least 2 starting points
      else:
         cu = 0.0
         if (m < 2):
             ifault[0] = 1

      if (cu > 0.0):
          alcu = log(cu)
      #set pointers
      iipt = 5
      iz = 8
      ihuz = nn+iz
      iscum = nn+ihuz
      ix = nn+iscum
      ihx = nn+ix
      ihpx = nn+ihx
      #store values in working vectors
      iwv[0] = ilow
      iwv[1] = ihigh
      iwv[2] = ns
      iwv[3] = 1
      if lb:
         iwv[4] = 1
      else:
         iwv[4] = 0

      if ub:
         iwv[5] = 1
      else:
         iwv[5] = 0

      if ( ns < m):
         ifault[0] = 2

      iwv[iipt+1] = 0
      rwv[0] = hulb
      rwv[1] = huub
      rwv[2] = emax
      rwv[3] = eps
      rwv[4] = cu
      rwv[5] = alcu
      rwv[6] = huzmax
      rwv[7] = xlb
      rwv[8] = xub
      rwv[iscum+1] = 1.0
      for i from 0 <= i < m:
         rwv[ix+i] = x[i]
         rwv[ihx+i] = hx[i]
         rwv[ihpx+i] = hpx[i]
      #create lower and upper hulls
      i = 0
      while (i < m):
            update(<int>iwv[3], <int>iwv[0], <int>iwv[1], &iwv[iipt+1], &rwv[iscum+1], rwv[4],
                    &rwv[ix+1], &rwv[ihx+1], &rwv[ihpx+1], &rwv[iz+1],
                    &rwv[ihuz+1], rwv[6], rwv[2], lb, rwv[7], rwv[0], ub,
                    rwv[8], rwv[1], ifault, rwv[3], rwv[5])
            i = <int>iwv[3]
            if (ifault[0] != 0):
               return

      #test for wrong starting points
      if ((not lb) and (hpx[<int>iwv[0]] < eps)):
         ifault[0] = 3
      if ((not ub) and (hpx[<int>iwv[1]] > -eps)):
         ifault[0] = 4
      return


cdef void sample(double* iwv, double* rwv, object h, object hprima,
        double* beta, int* ifault):
      """
      ifault
      0:successful sampling
      5:non-concavity detected
      6:random number generator generated zero
      7:numerical instability
      """
      cdef int iipt, iz, ns, nn, ihuz, iscum, ix, ihx, ihpx
      cdef int ub, lb

      #set pointers
      iipt = 5
      iz = 8
      ns = <int>iwv[2]
      nn = ns+1
      ihuz = nn+iz
      iscum = nn+ihuz
      ix = nn+iscum
      ihx = nn+ix
      ihpx = nn+ihx
      lb = 0
      ub = 0
      if (iwv[4] == 1):
         lb = 1
      if (iwv[5] == 1):
         ub = 1

      #call sampling subroutine
      spl1(ns, <int>iwv[3], <int>iwv[0], <int>iwv[1], &iwv[iipt+1], &rwv[iscum+1], rwv[4],
              &rwv[ix+1], &rwv[ihx+1], &rwv[ihpx+1], &rwv[iz+1], &rwv[ihuz+1],
              rwv[6], lb, rwv[7], rwv[0], ub, rwv[8], rwv[1], h, hprima, beta,
              ifault, rwv[2], rwv[3], rwv[5])
      return

cdef void spl1(int ns, int n, int ilow, int ihigh, double* ipt, double* scum,
        double cu, double* x, double* hx, double* hpx, double* z, double* huz,
        double huzmax, int lb, double xlb, double hulb, int ub, double xub,
        double huub, object h, object hprima, double* beta, int* ifault, double
        emax, double eps, double alcu):
     """
     this subroutine performs the adaptive rejection sampling, it calls
     subroutine splhull to sample from the upper hull, if the sampling
     involves a function evaluation it calls the updating subroutine
     ifault is a diagnostic of any problem: non concavity, 0 random number
     or numerical imprecision
     """
     cdef int i, j, n1
     cdef bint sampld
     cdef double u1, u2, alu1, fx
     cdef double alhl, alhu
     cdef int max_attempt = 3*ns
     sampld = False
     ifault[0] = 0
     cdef int attempts = 0
     while ((not sampld) and (attempts < max_attempt)):
         u2 = rand()/RAND_MAX
         #test for zero random number
         if (u2 == 0.0):
            ifault[0] = 6
            return
         splhull(u2, ipt, ilow, lb, xlb, hulb, huzmax, alcu, &x[0], &hx[0], &hpx[0], &z[0], &huz[0], &scum[0], eps, emax, beta, i, j)
         #sample u1 to compute rejection
         u1 = rand()/RAND_MAX
         if (u1 == 0.0):
            ifault[0] = 6
         alu1 = log(u1)
         # compute alhu: upper hull at point u1
         alhu = hpx[i]*(beta[0]-x[i])+hx[i]-huzmax
         if ((beta[0] > x[ilow]) and (beta[0] < x[ihigh])):
            # compute alhl: value of the lower hull at point u1
            if (beta[0] > x[i]):
               j = i
               i = <int>ipt[i]
            alhl = hx[i]+(beta[0]-x[i])*(hx[i]-hx[i])/(x[i]-x[i])-huzmax
            #squeezing test
            if ((alhl-alhu) > alu1):
               sampld = True
            #if not sampled evaluate the function, do the rejection test and update
         if (not sampld):
            n1 = n+1
            x[n1] = beta[0]
            hx[n1]=h(x[n1])
            hpx[n1] = hprima(x[n1])
            fx = hx[n1]-huzmax
            if (alu1 < (fx-alhu)):
               sampld = True
            # update while the number of points defining the hulls is lower than ns
            if (n < ns):
               update(n, ilow, ihigh, &ipt[0], &scum[0], cu, &x[0], &hx[0], &hpx[0], &z[0], &huz[0], huzmax, emax, lb, xlb, hulb, ub, xub, huub, ifault, eps, alcu)
            if (ifault[0] != 0):
               return
         attempts += 1
     if (attempts >= max_attempt):
       raise ValueError("Trap in ARS: Maximum number of attempts reached by routine spl1_\n")
     return

cdef void splhull(double u2, double* ipt, int ilow,
        int lb, double xlb, double hulb, double huzmax, double alcu,
        double* x, double* hx, double* hpx,
        double* z, double* huz, double* scum, double eps,
        double emax, double* beta, int i, int j):
      #this subroutine samples beta from the normalised upper hull
      cdef double eh, logdu, logtg, sign
      cdef bint horiz
      #
      i = ilow
      #
      #find from which exponential piece you sample
      while (u2 > scum[i]):
        j = i
        i = <int>ipt[i]

      if (i==ilow):
        #sample below z(ilow), depending on the existence of a lower bound
        if (lb) :
          eh = hulb-huzmax-alcu
          horiz = (fabs(hpx[ilow]) < eps)
          if (horiz):
             beta[0] = xlb+u2*expon(-eh, emax)
          else:
             sign = fabs(hpx[i])/hpx[i]
             logtg = log(fabs(hpx[i]))
             logdu = log(u2)
             eh = logdu+logtg-eh
             if (eh < emax):
                beta[0] = xlb+log(1.0+sign*expon(eh, emax))/hpx[i]
             else:
                beta[0] = xlb+eh/hpx[i]
        else:
          #hpx(i) must be positive, x(ilow) is left of the mode
          beta[0] = (log(hpx[i]*u2)+alcu-hx[i]+x[i]*hpx[i]+huzmax)/hpx[i]

      else:
        #sample above(j)
        eh = huz[j]-huzmax-alcu
        horiz = (fabs(hpx[i]) < eps)
        if (horiz):
           beta[0] = z[j]+(u2-scum[j])*expon(-eh, emax)
        else:
            sign = fabs(hpx[i])/hpx[i]
            logtg = log(fabs(hpx[i]))
            logdu = log(u2-scum[j])
            eh = logdu+logtg-eh
            if (eh < emax):
              beta[0] = z[j]+(log(1.0+sign*expon(eh, emax)))/hpx[j]
            else:
              beta[0] = z[j]+eh/hpx[j]
      return

cdef void intersection(double x1, double y1, double yp1, double x2, double y2,
        double yp2, double z1, double hz1, double eps, int* ifault):
     """
     computes the intersection (z1, hz1) between 2 tangents defined by
     x1, y1, yp1 and x2, y2, yp2
     """
     cdef double y12, y21, dh
     # first test for non-concavity
     y12 = y1+yp1*(x2-x1)
     y21 = y2+yp2*(x1-x2)
     if ((y21 < y1) or (y12 < y2)):
         ifault[0] = 5
         return

     dh = yp2-yp1
     #IF the lines are nearly parallel,
     #the intersection is taken at the midpoint
     if (fabs(dh) <= eps):
        z1 = 0.5*(x1+x2)
        hz1 = 0.5*(y1+y2)
     #Else compute from the left or the right for greater numerical precision
     elif (fabs(yp1) < fabs(yp2)):
        z1 = x2+(y1-y2+yp1*(x2-x1))/dh
        hz1 = yp1*(z1-x1)+y1
     else:
        z1 = x1+(y1-y2+yp2*(x2-x1))/dh
        hz1 = yp2*(z1-x2)+y2

     #test for misbehaviour due to numerical imprecision
     if ((z1 < x1) or (z1 > x2)):
        ifault[0] = 7
     return

cdef void update(int n, int ilow, int ihigh, double* ipt, double* scum, double
        cu, double* x, double* hx, const double* hpx, double* z, double* huz,
        double huzmax, double emax, int lb, double xlb, double hulb, int ub,
        double xub, double huub, int* ifault, double eps, double alcu):
      """
       this subroutine increments n and updates all the parameters which
       define the lower and the upper hull
      """
      cdef int i, j
      cdef bint horiz
      cdef double dh, u
      cdef double zero = 1e-2
      """

      DESCRIPTION OF PARAMETERS and place of storage

      ilow iwv[0]    : index of the smallest x(i)
      ihigh iwv[1]   : index of the largest x(i)
      n    iwv[3]    : number of points defining the hulls
      ipt  iwv[iipt] : pointer array:  ipt(i) is the index of the x(.)
                       immediately larger than x(i)
      hulb rwv[0]    : value of the upper hull at xlb
      huub rwv[1]    : value of the upper hull at xub
      cu   rwv[4]    : integral of the exponentiated upper hull divided
                       by exp(huzmax)
      alcu rwv[5]    : logarithm of cu
      huzmax rwv[6]  : maximum of huz(i); i = 1, n
      z    rwv[iz+1] : z(i) is the abscissa of the intersection between
                       the tangents at x(i) and x(ipt(i))
      huz  rwv[ihuz+1]: huz(i) is the ordinate of the intersection
                         defined above
      scum rwv[iscum]: scum(i) is the cumulative probability of the
                       normalised exponential of the upper hull
                       calculated at z(i)
      eps  rwv[3]    : =exp(-emax) a very small number
      """
      n = n+1
      #update z, huz and ipt
      if (x[n] < x[ilow]):
         #insert x(n) below x(ilow)
         #test for non-concavity
         if (hpx[ilow] > hpx[n]):
             ifault[0] = 5
         ipt[n]=ilow
         intersection(x[n], hx[n], hpx[n], x[ilow], hx[ilow], hpx[ilow], z[n], huz[n], eps, ifault)
         if (ifault[0] != 0):
             return
         if (lb):
            hulb = hpx[n]*(xlb-x[n])+hx[n]
         ilow = n
      else:
        i = ilow
        j = i
        #find where to insert x(n)
        while ((x[n]>=x[i]) and (ipt[i] != 0)):
          j = i
          i = <int>ipt[i]
        if (x[n] > x[i]):
           # insert above x(ihigh)
           # test for non-concavity
           if (hpx[i] < hpx[n]):
              ifault[0] = 5
           ihigh = n
           ipt[i] = n
           ipt[n] = 0
           intersection(x[i], hx[i], hpx[i], x[n], hx[n], hpx[n], z[i], huz[i], eps, ifault)
           if (ifault[0] != 0):
              return
           huub = hpx[n]*(xub-x[n])+hx[n]
           z[n] = 0.0
           huz[n] = 0.0
        else:
           # insert x(n) between x(j) and x(i)
           # test for non-concavity
           if ((hpx[j] < hpx[n]) or (hpx[i] > hpx[n])):
              ifault[0] = 5
           ipt[j]=n
           ipt[n]=i
           # insert z(j) between x(j) and x(n)
           intersection(x[j], hx[j], hpx[j], x[n], hx[n], hpx[n], z[j], huz[j], eps, ifault)
           if (ifault[0] != 0):
              return
           #insert z(n) between x(n) and x(i)
           intersection(x[n], hx[n], hpx[n], x[i], hx[i], hpx[i], z[n], huz[n], eps, ifault)
           if (ifault[0] != 0):
              return
      #update huzmax
      j = ilow
      i = <int>ipt[j]
      huzmax = huz[j]
      while ((huz[j] < huz[i]) and (ipt[i] != 0)):
        j = i
        i = <int>ipt[i]
        huzmax = max(huzmax, huz[j])
      if (lb):
          huzmax = max(huzmax, hulb)
      if (ub):
          huzmax = max(huzmax, huub)
      #update cu
      #scum receives area below exponentiated upper hull left of z(i)
      i = ilow
      horiz = (fabs(hpx[ilow]) < eps)
      if ((not lb) and (not horiz)):
        cu = expon(huz[i]-huzmax, emax)/hpx[i]
      elif (lb and horiz):
        cu = (z[ilow]-xlb)*expon(hulb-huzmax, emax)
      elif (lb and (not horiz)):
        dh = hulb-huz[i]
        if (dh > emax):
          cu = -expon(hulb-huzmax, emax)/hpx[i]
        else:
          cu = expon(huz[i]-huzmax, emax)*(1-expon(dh, emax))/hpx[i]
      else:
        cu = 0
      scum[i]=cu
      j = i
      i = <int>ipt[i]
      cdef int control_count = 0
      while (ipt[i] != 0):
        if (control_count > n):
          raise ValueError('Trap in ARS: infinite while in update near ...\n')
        control_count += 1
        dh = huz[j]-huz[i]
        horiz = (fabs(hpx[i]) < eps)
        if (horiz):
          cu += (z[i]-z[j])*expon((huz[i]+huz[j])*0.5-huzmax, emax)
        else:
          if (dh < emax):
            cu += expon(huz[i]-huzmax, emax)*(1-expon(dh, emax))/hpx[i]
          else:
            cu -= expon(huz[j]-huzmax, emax)/hpx[i]
        j = i
        i = <int>ipt[i]
        scum[j]=cu
      horiz = (fabs(hpx[i]) < eps)
      #if the derivative is very small the tangent is nearly horizontal
      if (not(ub or horiz)):
         cu -= expon(huz[j]-huzmax, emax)/hpx[i]
      elif (ub and horiz):
         cu += (xub-x[i])*expon((huub+hx[i])*0.5-huzmax, emax)
      elif (ub and (not horiz)):
         dh = huz[j]-huub
         if (dh > emax):
            cu -= expon(huz[j]-huzmax, emax)/hpx[i]
         else:
            cu += expon(huub-huzmax, emax)*(1-expon(dh, emax))/hpx[i]
      scum[i]=cu
      if (cu > 0):
         alcu = log(cu)
      #normalize scum to obtain a cumulative probability while excluding
      #unnecessary points
      i = ilow
      u = (cu-scum[i])/cu
      if ((u == 1.0) and (hpx[<int>ipt[i]] > zero)):
        ilow = <int>ipt[i]
        scum[i] = 0.0
      else:
        scum[i] = 1.0-u
      j = i
      i = <int>ipt[i]
      while (ipt[i] != 0):
        j = i
        i = <int>ipt[i]
        u = (cu-scum[j])/cu
        if ((u == 1.0) and (hpx[i] > zero)):
          ilow = i
        else:
          scum[j] = 1.0 - u
      scum[i] = 1.0
      if (ub):
          huub = hpx[ihigh]*(xub-x[ihigh])+hx[ihigh]
      if (lb):
          hulb = hpx[ilow]*(xlb-x[ilow])+hx[ilow]
      return


cdef double expon(double x, double emax):
     #performs an exponential without underflow
     cdef double expon
     if (x < -emax):
        expon = 0.0
     else:
        expon = exp(x)
     return expon
 
def main(int ns, int m, double emax,
         np.ndarray[ndim=1, dtype=np.float64_t] x,
         np.ndarray[ndim=1, dtype=np.float64_t] hx,
         np.ndarray[ndim=1, dtype=np.float64_t] hpx,
         int num,
         func_h,
         func_hprima,
         lb = False,
         ub = False):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] iwv, rwv, sp
    
    # initializing arrays
    rwv = np.zeros(ns*6+15, dtype=np.float64)
    iwv = np.zeros(ns+7, dtype=np.float64)
    sp = np.zeros(num, dtype=np.float64)
    
    cdef double xlb = np.min(x)
    cdef double xub = np.max(x)

    cdef int ifault = 999
    cdef double beta = 999.
    
    initial(ns, m, emax,
            &x[0], # passing array by reference
            &hx[0], # passing array by reference
            &hpx[0], # passing array by reference
            int(lb), # transforming bool in int
            xlb,
            int(ub), # transforming bool in int
            xub, 
            &ifault, # passing integer variable by reference
            &iwv[0], # passing array by reference
            &rwv[0] # passing array by reference
            )

    cdef int i
    for i in range(num):
        beta = 999
        sample(
                &iwv[0], # passing array by reference
                &rwv[0], # passing array by reference
                func_h, # function
                func_hprima, # function derivative
                &beta, # passing double variable by reference
                &ifault # passing integer variable by reference
                )
        sp[i] = beta

    return sp

