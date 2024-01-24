import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import TableIndex, TableInterpolator, TabularRate, Tfactors
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
nnuc = 2

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1

names = []
names.append("n")
names.append("h1")

def to_composition(Y):
    """Convert an array of molar fractions to a Composition object."""
    from pynucastro import Composition, Nucleus
    nuclei = [Nucleus.from_cache(name) for name in names]
    comp = Composition(nuclei)
    for i, nuc in enumerate(nuclei):
        comp.X[nuc] = Y[i] * A[i]
    return comp

@jitclass([
    ("p__n", numba.float64),
    ("n__p", numba.float64),
])
class RateEval:
    def __init__(self):
        self.p__n = np.nan
        self.n__p = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)


b0 = -0.62173 
b1 = 0.22211e2
b2 = -0.72798e2
b3 = 0.11571e3
b4 = -0.11763e2
b5 = 0.45521e2
b6 = -3.7973 
b7 = 0.41266 
b8 = -0.026210
b9 = 0.87934e-3
b10 = -0.12016e-4
qpn = 2.8602

@numba.njit() 
def p__n(rate_eval, tf):  
    # p --> n
    rate=0
    #rate from https://arxiv.org/pdf/astro-ph/0408076.pdf appendix C
    if tf.T9>1.160451812:
      b=[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
      z=5.92989658*tf.T9i
      for i in range(11):
         rate+=1/880.2*np.exp(-qpn*z)*b[i]*z**-i 
        
    #Kawano rate
    #rate=1/879.6*(5.252/z - 16.229/z**2 + 18.059/z**3 + 34.181/z**4 + 27.617/z**5)*np.exp(-2.530988*z)
    rate_eval.p__n = rate


a0 = 1
a1 = 0.15735 
a2 = 4.6172
a3 = -0.40520e2 
a4 = 0.13875e3 
a5 = -0.59898e2
a6 = 0.66752e2 
a7 = -0.16705e2 
a8 = 3.8071
a9 = -0.39140 
a10 = 0.023590 
a11 = -0.83696e-4
a12 = -0.42095e-4 
a13 = 0.17675e-5
qnp = 0.33979 

@numba.njit()
def n__p(rate_eval, tf):
    # n --> p
    z=5.92989658*tf.T9i
    #rate from https://arxiv.org/pdf/astro-ph/0408076.pdf appendix C
    rate=0
    a=[a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13]
    for i in range(14):
      rate+=1/880.2*np.exp(-qnp/z)*a[i]*z**-i

    #Kawano rate
    #rate = 1/879.6*(0.565/z - 6.382/z**2 + 11.108/z**3 + 36.492/z**4 + 27.512/z**5)

    rate_eval.n__p = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates

    # custom rates
    p__n(rate_eval, tf)
    n__p(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -Y[jn]*rate_eval.n__p
       +Y[jp]*rate_eval.p__n
       )

    dYdt[jp] = (
       -Y[jp]*rate_eval.p__n
       +Y[jn]*rate_eval.n__p
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates

    # custom rates
    p__n(rate_eval, tf)
    n__p(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rate_eval.n__p
       )

    jac[jn, jp] = (
       +rate_eval.p__n
       )

    jac[jp, jn] = (
       +rate_eval.n__p
       )

    jac[jp, jp] = (
       -rate_eval.p__n
       )

    return jac

#For AoT compilation of the network
def AoT(networkname):
   from numba.pycc import CC

   cc = CC(networkname)
   # Uncomment the following line to print out the compilation steps
   #cc.verbose = True

   #
   @cc.export('nnuc','i4()')
   def nNuc():
      return nnuc

   @cc.export('rhs', 'f8[:](f8, f8[:], f8, f8)')
   def rhsCC(t, Y, rho, T):
      return rhs_eq(t, Y, rho, T, None)

   @cc.export('jacobian', '(f8, f8[:], f8, f8)')
   def jacobian(t, Y, rho, T):
      return jacobian_eq(t, Y, rho, T, None)


   cc.compile()
