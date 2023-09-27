import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import TableIndex, TableInterpolator, TabularRate, Tfactors
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
jd = 2
nnuc = 3

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jd] = 2

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jd] = 1

names = []
names.append("n")
names.append("h1")
names.append("h2")

def to_composition(Y):
    """Convert an array of molar fractions to a Composition object."""
    from pynucastro import Composition, Nucleus
    nuclei = [Nucleus.from_cache(name) for name in names]
    comp = Composition(nuclei)
    for i, nuc in enumerate(nuclei):
        comp.X[nuc] = Y[i] * A[i]
    return comp

@jitclass([
    ("d__n_p", numba.float64),
    ("n_p__d", numba.float64),
    ("p_p__d__weak__bet_pos_", numba.float64),
    ("p_p__d__weak__electron_capture", numba.float64),
    ("p_d__n_p_p", numba.float64),
    ("n_p_p__p_d", numba.float64),
    ("p__n", numba.float64),
    ("n__p", numba.float64),
])
class RateEval:
    def __init__(self):
        self.d__n_p = np.nan
        self.n_p__d = np.nan
        self.p_p__d__weak__bet_pos_ = np.nan
        self.p_p__d__weak__electron_capture = np.nan
        self.p_d__n_p_p = np.nan
        self.n_p_p__p_d = np.nan
        self.p__n = np.nan
        self.n__p = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def d__n_p(rate_eval, tf):
    # d --> n + p
    rate = 0.0

    # an06n
    rate += np.exp(  33.0154 + -25.815*tf.T9i + -2.30472*tf.T913
                  + -0.887862*tf.T9 + 0.137663*tf.T953 + 1.5*tf.lnT9)
    # an06n
    rate += np.exp(  34.6293 + -25.815*tf.T9i + -2.70618*tf.T913
                  + 0.11718*tf.T9 + -0.00312788*tf.T953 + 1.96913*tf.lnT9)
    # an06n
    rate += np.exp(  31.1075 + -25.815*tf.T9i + -0.0102082*tf.T913
                  + -0.0893959*tf.T9 + 0.00696704*tf.T953 + 2.5*tf.lnT9)

    rate_eval.d__n_p = rate

@numba.njit()
def n_p__d(rate_eval, tf):
    # n + p --> d
    rate = 0.0

    # an06n
    rate += np.exp(  12.3687 + -2.70618*tf.T913
                  + 0.11718*tf.T9 + -0.00312788*tf.T953 + 0.469127*tf.lnT9)
    # an06n
    rate += np.exp(  10.7548 + -2.30472*tf.T913
                  + -0.887862*tf.T9 + 0.137663*tf.T953)
    # an06n
    rate += np.exp(  8.84688 + -0.0102082*tf.T913
                  + -0.0893959*tf.T9 + 0.00696704*tf.T953 + 1.0*tf.lnT9)

    rate_eval.n_p__d = rate

@numba.njit()
def p_p__d__weak__bet_pos_(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    # bet+w
    rate += np.exp(  -34.7863 + -3.51193*tf.T913i + 3.10086*tf.T913
                  + -0.198314*tf.T9 + 0.0126251*tf.T953 + -1.02517*tf.lnT9)

    rate_eval.p_p__d__weak__bet_pos_ = rate

@numba.njit()
def p_p__d__weak__electron_capture(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    #   ecw
    rate += np.exp(  -43.6499 + -0.00246064*tf.T9i + -2.7507*tf.T913i + -0.424877*tf.T913
                  + 0.015987*tf.T9 + -0.000690875*tf.T953 + -0.207625*tf.lnT9)

    rate_eval.p_p__d__weak__electron_capture = rate

@numba.njit()
def p_d__n_p_p(rate_eval, tf):
    # d + p --> n + p + p
    rate = 0.0

    # cf88n
    rate += np.exp(  17.3271 + -25.82*tf.T9i + -3.72*tf.T913i + 0.946313*tf.T913
                  + 0.105406*tf.T9 + -0.0149431*tf.T953)

    rate_eval.p_d__n_p_p = rate

@numba.njit()
def n_p_p__p_d(rate_eval, tf):
    # n + p + p --> p + d
    rate = 0.0

    # cf88n
    rate += np.exp(  -4.24034 + -3.72*tf.T913i + 0.946313*tf.T913
                  + 0.105406*tf.T9 + -0.0149431*tf.T953 + -1.5*tf.lnT9)

    rate_eval.n_p_p__p_d = rate


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
    z=5.92989658*tf.T9i
    rate=0
    #rate from https://arxiv.org/pdf/astro-ph/0408076.pdf appendix C
    b=[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
    if tf.T9>1.160451812:
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
    d__n_p(rate_eval, tf)
    n_p__d(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)

    # custom rates
    p__n(rate_eval, tf)
    n__p(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p__d__weak__bet_pos_ *= scor
        rate_eval.p_p__d__weak__electron_capture *= scor
        rate_eval.n_p_p__p_d *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d__n_p_p *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -Y[jn]*rate_eval.n__p
       +Y[jd]*rate_eval.d__n_p
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +Y[jp]*rate_eval.p__n
       )

    dYdt[jp] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -Y[jp]*rate_eval.p__n
       +Y[jd]*rate_eval.d__n_p
       +2*rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       +Y[jn]*rate_eval.n__p
       )

    dYdt[jd] = (
       -Y[jd]*rate_eval.d__n_p
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    d__n_p(rate_eval, tf)
    n_p__d(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)

    # custom rates
    p__n(rate_eval, tf)
    n__p(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p__d__weak__bet_pos_ *= scor
        rate_eval.p_p__d__weak__electron_capture *= scor
        rate_eval.n_p_p__p_d *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d__n_p_p *= scor

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       -rate_eval.n__p
       )

    jac[jn, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       +rho*Y[jd]*rate_eval.p_d__n_p_p
       +rate_eval.p__n
       )

    jac[jn, jd] = (
       +rate_eval.d__n_p
       +rho*Y[jp]*rate_eval.p_d__n_p_p
       )

    jac[jp, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       +rate_eval.n__p
       )

    jac[jp, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       -rate_eval.p__n
       +2*rho*Y[jd]*rate_eval.p_d__n_p_p
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       )

    jac[jp, jd] = (
       -rho*Y[jp]*rate_eval.p_d__n_p_p
       +rate_eval.d__n_p
       +2*rho*Y[jp]*rate_eval.p_d__n_p_p
       )

    jac[jd, jn] = (
       +rho*Y[jp]*rate_eval.n_p__d
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       )

    jac[jd, jp] = (
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jn]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       )

    jac[jd, jd] = (
       -rate_eval.d__n_p
       -rho*Y[jp]*rate_eval.p_d__n_p_p
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
