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
    double RAND_MAX
    double c_libc_random "random"()
    void c_libc_srandom "srandom"(unsigned int seed)
     
from cython.parallel import prange

from libc.math cimport fabs

ctypedef void (*func_t)(double *, double *, double *) 
cdef void initial(int *ns, int *m, double *emax, double* x, double* hx, double*
        hpx, int *lb, double *xlb, int *ub, double *xub, int* ifault, int* iwv,
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
      eps = expon(-(emax[0]), emax[0])
      ifault[0] = 0
      ilow = 0
      ihigh = 0
      nn = ns[0]+1
      #at least one starting point
      if (m[0] < 1):
         ifault[0] = 1

      huzmax = hx[0]
      if not ub[0]:
         xub[0] = 0.0

      if not lb[0]:
         xlb[0] = 0.0

      hulb = (xlb[0]-x[0])*hpx[0] + hx[0]
      huub = (xub[0]-x[0])*hpx[0] + hx[0]
      #if bounded on both sides
      if (ub[0] and lb[0]):
         huzmax = max(huub, hulb)
         horiz = (fabs(hpx[0]) < eps)
         if (horiz):
           cu = expon((huub+hulb)*0.5-huzmax, emax[0])*(xub[0]-xlb[0])
         else:
           cu = expon(huub-huzmax, emax[0])*(1-expon(hulb-huub, emax[0]))/hpx[0]
      elif ((ub[0]) and (not lb[0])):
         #if bounded on the right and unbounded on the left
         huzmax = huub
         cu = 1.0/hpx[0]

      elif ((not ub[0]) and (lb[0])):
         #if bounded on the left and unbounded on the right
         huzmax = hulb
         cu = -1.0/hpx[0]

         #if unbounded at least 2 starting points
      else:
         cu = 0.0
         if (m[0] < 2):
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
      iwv[2] = ns[0]
      iwv[3] = 0
      if lb[0]:
         iwv[4] = 1
      else:
         iwv[4] = 0

      if ub[0]:
         iwv[5] = 1
      else:
         iwv[5] = 0

      if ( ns[0] < m[0]):
         ifault[0] = 2

      iwv[iipt+1] = 0
      rwv[0] = hulb
      rwv[1] = huub
      rwv[2] = emax[0]
      rwv[3] = eps
      rwv[4] = cu
      rwv[5] = alcu
      rwv[6] = huzmax
      rwv[7] = xlb[0]
      rwv[8] = xub[0]
      rwv[iscum] = 1.0
      for i from 0 <= i < m[0]:
         rwv[ix+i] = x[i]
         rwv[ihx+i] = hx[i]
         rwv[ihpx+i] = hpx[i]
      #create lower and upper hulls
      i = 0
      while (i < (m[0]-1)):
            update(&iwv[3], &iwv[0], &iwv[1], &iwv[iipt+1], &rwv[iscum], &rwv[4],
                    &rwv[ix], &rwv[ihx], &rwv[ihpx], &rwv[iz+1],
                    &rwv[ihuz+1], &rwv[6], &rwv[2], lb, &rwv[7], &rwv[0], ub,
                    &rwv[8], &rwv[1], ifault, &rwv[3], &rwv[5])
            i = iwv[3]
            if (ifault[0] != 0):
               return
      print "UPDATE FINISHED!!!"
      #test for wrong starting points
      if ((not lb[0]) and (hpx[iwv[0]] < eps)):
         ifault[0] = 3
      if ((not ub[0]) and (hpx[iwv[1]] > -eps)):
         ifault[0] = 4
      return


cdef void sample(int* iwv, double* rwv, func_t f,
        double* beta, int* ifault):
      """
      ne: number of elements of pointer x
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
      ns = iwv[2]
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
      spl1(&ns, &iwv[3], &iwv[0], &iwv[1], &iwv[iipt+1], &rwv[iscum+1], &rwv[4],
              &rwv[ix+1], &rwv[ihx+1], &rwv[ihpx+1], &rwv[iz+1], &rwv[ihuz+1],
              &rwv[6], &lb, &rwv[7], &rwv[0], &ub, &rwv[8], &rwv[1], f, beta,
              ifault, &rwv[2], &rwv[3], &rwv[5])
      return


    
cdef void spl1(int *ns, int *n, int *ilow, int *ihigh, int* ipt, double* scum,
        double *cu, double* x, double* hx, double* hpx, double* z, double* huz,
        double *huzmax, int *lb, double *xlb, double *hulb, int *ub, double *xub,
        double *huub, func_t f, double* beta, int* ifault, double
        *emax, double *eps, double *alcu):
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
     cdef int max_attempt = 3*ns[0]
     sampld = False
     ifault[0] = 0
     cdef int attempts = 0
     cdef double rm = RAND_MAX
     while ((not sampld) and (attempts < max_attempt)):
         
         u2 = c_libc_random()/rm
         print "u2:",u2
         #test for zero random number
         if (u2 == 0.0):
            ifault[0] = 6
            return
         splhull(&u2, &ipt[0], ilow, lb, xlb, hulb, huzmax, alcu, &x[0], &hx[0], &hpx[0], &z[0], &huz[0], &scum[0], eps, emax, beta, &i, &j)
         #sample u1 to compute rejection
         u1 = c_libc_random()/rm
         if (u1 == 0.0):
            ifault[0] = 6
         alu1 = log(u1)
         # compute alhu: upper hull at point u1
         alhu = hpx[i]*(beta[0]-x[i])+hx[i]-huzmax[0]
         if ((beta[0] > x[ilow[0]]) and (beta[0] < x[ihigh[0]])):
            # compute alhl: value of the lower hull at point u1
            if (beta[0] > x[i]):
               j = i
               i = ipt[i]
            alhl = hx[i]+(beta[0]-x[i])*(hx[i]-hx[j])/(x[i]-x[j])-huzmax[0]
            #squeezing test
            if ((alhl-alhu) > alu1):
               
               sampld = True
         print "update"
         #if not sampled evaluate the function, do the rejection test and update
         if (not sampld):
            n1 = n[0]+1
            x[n1] = beta[0]
            #defining log of the distribution and its derivitive
            print "compute values at distributions"
            print hx[n1], hx[1],x[1]
            f(&x[n1], &hx[n1], &hpx[n1])
            fx = hx[n1]-huzmax[0]
            print hx[n1]
            if (alu1 < (fx-alhu)):
               sampld = True
            # update while the number of points defining the hulls is lower than ns
            if (n[0] < ns[0]):
               update(n, ilow, ihigh, &ipt[0], &scum[0], cu, &x[0], &hx[0], &hpx[0], &z[0], &huz[0], huzmax, emax, lb, xlb, hulb, ub, xub, huub, ifault, eps, alcu)
            if (ifault[0] != 0):
               return
         attempts += 1
     if (attempts >= max_attempt):
       raise ValueError("Trap in ARS: Maximum number of attempts reached by routine spl1_\n")
     return

cdef void splhull(double *u2, int* ipt, int *ilow,
        int *lb, double *xlb, double *hulb, double *huzmax, double *alcu,
        double* x, double* hx, double* hpx,
        double* z, double* huz, double* scum, double *eps,
        double *emax, double* beta, int *i, int *j):
      #this subroutine samples beta from the normalised upper hull
      cdef double eh, logdu, logtg, sign
      cdef bint horiz
      #
      i[0] = ilow[0]
      #
      #find from which exponential piece you sample
      while (u2[0] > scum[i[0]]):
        j[0] = i[0]
        i[0] = <int>ipt[i[0]]

      if (i[0]==ilow[0]):
        #sample below z(ilow), depending on the existence of a lower bound
        if (lb[0]) :
          eh = hulb[0]-huzmax[0]-alcu[0]
          horiz = (fabs(hpx[ilow[0]]) < eps[0])
          if (horiz):
             beta[0] = xlb[0]+u2[0]*expon(-eh, emax[0])
          else:
             sign = fabs(hpx[i[0]])/hpx[i[0]]
             logtg = log(fabs(hpx[i[0]]))
             logdu = log(u2[0])
             eh = logdu+logtg-eh
             if (eh < emax[0]):
                beta[0] = xlb[0]+log(1.0+sign*expon(eh, emax[0]))/hpx[i[0]]
             else:
                beta[0] = xlb[0]+eh/hpx[i[0]]
        else:
          #hpx(i) must be positive, x(ilow) is left of the mode
          beta[0] = (log(hpx[i[0]]*u2[0])+alcu[0]-hx[i[0]]+x[i[0]]*hpx[i[0]]+huzmax[0])/hpx[i[0]]

      else:
        #sample above(j)
        eh = huz[j[0]]-huzmax[0]-alcu[0]
        horiz = (fabs(hpx[i[0]]) < eps[0])
        if (horiz):
           beta[0] = z[j[0]]+(u2[0]-scum[j[0]])*expon(-eh, emax[0])
        else:
            sign = fabs(hpx[i[0]])/hpx[i[0]]
            logtg = log(fabs(hpx[i[0]]))
            logdu = log(u2[0]-scum[j[0]])
            eh = logdu+logtg-eh
            if (eh < emax[0]):
              beta[0] = z[j[0]]+(log(1.0+sign*expon(eh, emax[0])))/hpx[i[0]]
            else:
              beta[0] = z[j[0]]+eh/hpx[i[0]]
      return

cdef void intersection(double *x1, double *y1, double *yp1, double *x2, double *y2,
        double *yp2, double *z1, double *hz1, double *eps, int* ifault):
     """
     computes the intersection (z1, hz1) between 2 tangents defined by
     x1, y1, yp1 and x2, y2, yp2
     """
     cdef double y12, y21, dh
     # first test for non-concavity
     y12 = y1[0]+yp1[0]*(x2[0]-x1[0])
     y21 = y2[0]+yp2[0]*(x1[0]-x2[0])
     if ((y21 < y1[0]) or (y12 < y2[0])):
         ifault[0] = 5
         return

     dh = yp2[0]-yp1[0]
     #IF the lines are nearly parallel,
     #the intersection is taken at the midpoint
     if (fabs(dh) <= eps[0]):
        z1[0] = 0.5*(x1[0]+x2[0])
        hz1[0] = 0.5*(y1[0]+y2[0])
     #Else compute from the left or the right for greater numerical precision
     elif (fabs(yp1[0]) < fabs(yp2[0])):
        z1[0] = x2[0]+(y1[0]-y2[0]+yp1[0]*(x2[0]-x1[0]))/dh
        hz1[0] = yp1[0]*(z1[0]-x1[0])+y1[0]
     else:
        z1[0] = x1[0]+(y1[0]-y2[0]+yp2[0]*(x2[0]-x1[0]))/dh
        hz1[0] = yp2[0]*(z1[0]-x2[0])+y2[0]

     #test for misbehaviour due to numerical imprecision
     if ((z1[0] < x1[0]) or (z1[0] > x2[0])):
        ifault[0] = 7
     return

cdef void update(int *n, int *ilow, int *ihigh, int* ipt, double* scum, double
        *cu, double* x, double* hx, double* hpx, double* z, double* huz,
        double *huzmax, double *emax, int *lb, double *xlb, double *hulb, int *ub,
        double *xub, double *huub, int* ifault, double *eps, double *alcu):
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
      n[0] = n[0]+1
      #made a change here
      print "ipt:", ipt[0]
      print "scum: ", scum[0]
      print "number of points defining the hulls", n[0]
      print " values of x: " ,  x[n[0]]
      print "index of the smallest x(i)", ilow[0] 
      print " values of x: " ,x[ilow[0]]
      print "z: ", z[0]
      print "huz: ",huz[0]
      print "huzmax: ", huzmax[0]
      print "emax: ", emax[0]
      print "xlb: ", xlb[0]
      print "hulb: ", hulb[0]
      print "xub: ", xub[0]
      print "huub: ", huub[0]
      print "ifault: ", ifault[0] 
      print "eps: ", eps[0] 
      print "alcu: ", alcu[0]
      print "Update z,huz and ipt   "                              

      #update z, huz and ipt
      if (x[n[0]] < x[ilow[0]]):
         #insert x(n) below x(ilow)
         #test for non-concavity
         if (hpx[ilow[0]] > hpx[n[0]]):
             ifault[0] = 5
         ipt[n[0]]=ilow[0]
         intersection(&x[n[0]], &hx[n[0]], &hpx[n[0]], &x[ilow[0]], &hx[ilow[0]], &hpx[ilow[0]], &z[n[0]], &huz[n[0]], eps, ifault)
         print "insert x(n) below x(ilow)"
         print "value of x, z at n :", x[n[0]], z[n[0]], huz[n[0]]
         if (ifault[0] != 0):
             return
         if (lb[0]):
            hulb[0] = hpx[n[0]]*(xlb[0]-x[n[0]])+hx[n[0]]
         ilow[0] = n[0]
      else:
        i = ilow[0]
        j = i
        #find where to insert x(n)
        print "Check: ", x[n[0]],x[i]
        while ((x[n[0]]>=x[i]) and (ipt[i] != 0)):
          j = i
          i = <int>ipt[i]
          print "find where to insert x(n) : ", i, ilow[0]
        if (x[n[0]] >= x[i]):
           # insert above x(ihigh)
           # test for non-concavity
           if (hpx[i] < hpx[n[0]]):
              print "Trap: non-logcocavity detected by ARS update function\nhpx[i]=%e, hpx[n]=%e\n"%(hpx[i], hpx[n[0]])
              ifault[0] = 5
           ihigh[0] = n[0]
           ipt[i] = n[0]
           ipt[n[0]] = 0
           intersection(&x[i], &hx[i], &hpx[i], &x[n[0]], &hx[n[0]], &hpx[n[0]], &z[i], &huz[i], eps, ifault)
           print "insert x(n) above x(ihigh) " 
           print "value of z at i :" ,i ,z[i]  
           print "value of x at n:", n[0], x[n[0]]
           if (ifault[0] != 0):
              return
           huub[0] = hpx[n[0]]*(xub[0]-x[n[0]])+hx[n[0]]
           z[n[0]] = 0.0
           huz[n[0]] = 0.0
        else:
           # insert x(n) between x(j) and x(i)
           # test for non-concavity
           if ((hpx[j] < hpx[n[0]]) or (hpx[i] > hpx[n[0]])):
              print "Trap: non-logcocavity detected by ARS update_ function\nhpx[j]=%e, hpx[i]=%e, hpx[n]=%e\n"(hpx[j], hpx[i], hpx[n[0]]) 
              ifault[0] = 5
           ipt[j]=n[0]
           ipt[n[0]]=i
           # insert z(j) between x(j) and x(n)
           intersection(&x[j], &hx[j], &hpx[j], &x[n[0]], &hx[n[0]], &hpx[n[0]], &z[j], &huz[j], eps, ifault)
           print "insert z(j) between x(j) and x(n)" 
           print j, n[0], z[j], x[j], x[n[0]]
           if (ifault[0] != 0):
              return
           #insert z(n) between x(n) and x(i)
           intersection(&x[n[0]], &hx[n[0]], &hpx[n[0]], &x[i], &hx[i], &hpx[i], &z[n[0]], &huz[n[0]], eps, ifault)
           print "insert z(n) between x(n) and x(i)"       
           print n[0], i , z[n[0]] , x[n[0]], x[i]          
           if (ifault[0] != 0):
              return
      #update huzmax
      j = ilow[0]
      # made change here
      print "Update huzmax....."
      print "the index of the x(.) ",ipt[j] 
      print "indexes of the highest and smallest", ihigh[0], ilow[0]
      i = <int>ipt[j]
      huzmax[0] = huz[j]
      print "huzmax:",huzmax[0]                                                   
      while ((huz[j] < huz[i]) and (ipt[i] != 0)):
        j = i
        i = <int>ipt[i]
        huzmax[0] = max(huzmax[0], huz[j])
      print "maximum of huz(i) : ", huzmax[0] 
      print "value of i: ",i
      if (lb[0]):
          huzmax[0] = max(huzmax[0], hulb[0])
      if (ub[0]):
          huzmax[0] = max(huzmax[0], huub[0])
      #update cu
      #scum receives area below exponentiated upper hull left of z(i)
      i = ilow[0]
      horiz = (fabs(hpx[ilow[0]]) < eps[0])
      print "Update cu..."
      print "cu:", cu[0], horiz 
      if ((not lb[0]) and (not horiz)):
          print "NOT LOWER BOUND .."
          cu[0] = expon(huz[i]-huzmax[0], emax[0])/hpx[i]
      elif (lb[0] and horiz):
          print "LOWER BOUND .."
          cu[0] = (z[ilow[0]]-xlb[0])*expon(hulb[0]-huzmax[0], emax[0])
      elif (lb[0] and (not horiz)):
          dh = hulb[0]-huz[i]
          if (dh > emax[0]):
             print "dh:", dh
             cu[0] = -expon(hulb[0]-huzmax[0], emax[0])/hpx[i]
          else:
             print "huz:", huz[i]
             cu[0] = expon(huz[i]-huzmax[0], emax[0])*(1-expon(dh, emax[0]))/hpx[i]
      else:
        cu[0] = 0
      scum[i]=cu[0]
      print "scum:", scum[i] 
      j = i
      i = <int>ipt[i]
      print "i:", i, ", scum:", scum[i]
      print "the index of the x(.)" ,ipt[i]                                          
      cdef int control_count = 0
      while (ipt[i] != 0):
        if (control_count > n[0]):
          raise ValueError('Trap in ARS: infinite while in update near ...\n')
        control_count += 1
        dh = huz[j]-huz[i]
        horiz = (fabs(hpx[i]) < eps[0])
        if (horiz):
          cu[0] += (z[i]-z[j])*expon((huz[i]+huz[j])*0.5-huzmax[0], emax[0])
        else:
          if (dh < emax[0]):
            cu[0] += expon(huz[i]-huzmax[0], emax[0])*(1-expon(dh, emax[0]))/hpx[i]
          else:
            cu[0] -= expon(huz[j]-huzmax[0], emax[0])/hpx[i]
        j = i
        i = <int>ipt[i]
        scum[j]=cu[0]
      horiz = (fabs(hpx[i]) < eps[0])
      #if the derivative is very small the tangent is nearly horizontal
      if (not(ub[0] or horiz)):
         cu[0] -= expon(huz[j]-huzmax[0], emax[0])/hpx[i]
      elif (ub[0] and horiz):
         cu[0] += (xub[0]-x[i])*expon((huub[0]+hx[i])*0.5-huzmax[0], emax[0])
      elif (ub[0] and (not horiz)):
         dh = huz[j]-huub[0]
         if (dh > emax[0]):
            cu[0] -= expon(huz[j]-huzmax[0], emax[0])/hpx[i]
         else:
            cu[0] += expon(huub[0]-huzmax[0], emax[0])*(1-expon(dh, emax[0]))/hpx[i]
      scum[i]=cu[0]
      print "INDEX i: ", i 
      print "normalize scum at i :", scum[i]
      if (cu[0] > 0):
         alcu[0] = log(cu[0])
      #normalize scum to obtain a cumulative probability while excluding
      #unnecessary points
      i = ilow[0]
      print "CHANGE i VALUE"
      print "INDEX i (ilow): ", i
      print "normalize scum at i :", scum[i]                                                           
      u = (cu[0]-scum[i])/cu[0]
      print "normalization value :", u        
      if ((u == 1.0) and (hpx[<int>ipt[i]] > zero)):
        ilow[0] = <int>ipt[i]
        scum[i] = 0.0
      else:
        scum[i] = 1.0-u
      j = i
      i = <int>ipt[i]
      print "new value of i and ilow: ", i, ilow[0]
      while (ipt[i] != 0):
        j = i
        i = <int>ipt[i]
        u = (cu[0]-scum[j])/cu[0]
        if ((u == 1.0) and (hpx[i] > zero)):
          ilow[0] = i
        else:
          scum[j] = 1.0 - u
        print "ilow after all: ", i, ilow[0]
      scum[i] = 1.0
      if (ub[0]):
          huub[0] = hpx[ihigh[0]]*(xub[0]-x[ihigh[0]])+hx[ihigh[0]]
      if (lb[0]):
          hulb[0] = hpx[ilow[0]]*(xlb[0]-x[ilow[0]])+hx[ilow[0]]
      return


cdef double expon(double x, double emax):
     #performs an exponential without underflow
     cdef double expon
     if (x < -emax):
        expon = 0.0
     else:
        expon = exp(x)
     return expon

 
def normal(double[:] u,
           double[:] yu,
           double[:] ypu):          
     yu[0] = -u[0]*u[0]*0.5                                                               
     ypu[0]= -u[0] 
     return 


def normal_ctypes(u, yu, ypu):
   u_as_ctypes_array = (ctypes.c_double*1).from_address(ctypes.addressof(u.contents))
   yu_as_ctypes_array = (ctypes.c_double*1).from_address(ctypes.addressof(yu.contents))
   ypu_as_ctypes_array = (ctypes.c_double*1).from_address(ctypes.addressof(ypu.contents))
   normal(u_as_ctypes_array, yu_as_ctypes_array,ypu_as_ctypes_array)
     

def py_ars(int ns, int m, double emax,
           np.ndarray[ndim=1, dtype=np.float64_t] x,
           np.ndarray[ndim=1, dtype=np.float64_t] hx,
           np.ndarray[ndim=1, dtype=np.float64_t] hpx,
           int num,
           f #log of the distribution
           ):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] rwv, sp
    cdef np.ndarray[ndim=1, dtype=np.int64_t] iwv
    # initializing arrays
    rwv = np.zeros(ns*6+15, dtype=np.float64)
    iwv = np.zeros(ns+7, dtype=np.int64)
    sp = np.zeros(num, dtype=np.float64)
    
    cdef double xlb = np.min(x)
    cdef double xub = np.max(x)
    cdef int lb=0
    cdef int ub=0
    cdef int ifault = 999
    cdef double beta = 0.

    initial(&ns, &m, &emax,
            &x[0], # passing array by reference
            &hx[0], # passing array by reference
            &hpx[0], # passing array by reference
            &lb, # transforming bool in int
            &xlb,
            &ub, # transforming bool in int
            &xub, 
            &ifault, # passing integer variable by reference
            <int *>(&iwv[0]), # passing array by reference
            &rwv[0] # passing array by reference
            )
    FTYPE = ctypes.CFUNCTYPE(None, # return type
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double))
    f = FTYPE(f) # convert Python callable to ctypes function pointer

    # a rather nasty line to convert to a C function pointer
    cdef func_t f_ptr = (<func_t*><size_t>ctypes.addressof(f))[0]
    cdef int i
    for i from 0 <= i <num:
        sample(
               <int *>(&iwv[0]), # passing array by reference
               &rwv[0], # passing array by reference
               f_ptr,
               &beta, # passing double variable by reference
               &ifault, # passing integer variable by reference
               ) 
        sp[i] = beta

    return sp     

