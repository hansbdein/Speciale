import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import TableIndex, TableInterpolator, TabularRate, Tfactors
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
jd = 2
jt = 3
jhe3 = 4
jhe4 = 5
nnuc = 6

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jd] = 2
A[jt] = 3
A[jhe3] = 3
A[jhe4] = 4

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jd] = 1
Z[jt] = 1
Z[jhe3] = 2
Z[jhe4] = 2

names = []
names.append("n")
names.append("h1")
names.append("h2")
names.append("h3")
names.append("he3")
names.append("he4")

def to_composition(Y):
    """Convert an array of molar fractions to a Composition object."""
    from pynucastro import Composition, Nucleus
    nuclei = [Nucleus.from_cache(name) for name in names]
    comp = Composition(nuclei)
    for i, nuc in enumerate(nuclei):
        comp.X[nuc] = Y[i] * A[i]
    return comp

@jitclass([
    ("t__he3__weak__wc12", numba.float64),
    ("he3__t__weak__electron_capture", numba.float64),
    ("d__n_p", numba.float64),
    ("t__n_d", numba.float64),
    ("he3__p_d", numba.float64),
    ("he4__n_he3", numba.float64),
    ("he4__p_t", numba.float64),
    ("he4__d_d", numba.float64),
    ("n_p__d", numba.float64),
    ("p_p__d__weak__bet_pos_", numba.float64),
    ("p_p__d__weak__electron_capture", numba.float64),
    ("n_d__t", numba.float64),
    ("p_d__he3", numba.float64),
    ("d_d__he4", numba.float64),
    ("p_t__he4", numba.float64),
    ("n_he3__he4", numba.float64),
    ("p_he3__he4__weak__bet_pos_", numba.float64),
    ("d_d__n_he3", numba.float64),
    ("d_d__p_t", numba.float64),
    ("p_t__n_he3", numba.float64),
    ("p_t__d_d", numba.float64),
    ("d_t__n_he4", numba.float64),
    ("n_he3__p_t", numba.float64),
    ("n_he3__d_d", numba.float64),
    ("d_he3__p_he4", numba.float64),
    ("t_he3__d_he4", numba.float64),
    ("n_he4__d_t", numba.float64),
    ("p_he4__d_he3", numba.float64),
    ("d_he4__t_he3", numba.float64),
    ("p_d__n_p_p", numba.float64),
    ("t_t__n_n_he4", numba.float64),
    ("t_he3__n_p_he4", numba.float64),
    ("he3_he3__p_p_he4", numba.float64),
    ("n_p_p__p_d", numba.float64),
    ("n_n_he4__t_t", numba.float64),
    ("n_p_he4__t_he3", numba.float64),
    ("p_p_he4__he3_he3", numba.float64),
    ("p__n", numba.float64),
    ("n__p", numba.float64),
])
class RateEval:
    def __init__(self):
        self.t__he3__weak__wc12 = np.nan
        self.he3__t__weak__electron_capture = np.nan
        self.d__n_p = np.nan
        self.t__n_d = np.nan
        self.he3__p_d = np.nan
        self.he4__n_he3 = np.nan
        self.he4__p_t = np.nan
        self.he4__d_d = np.nan
        self.n_p__d = np.nan
        self.p_p__d__weak__bet_pos_ = np.nan
        self.p_p__d__weak__electron_capture = np.nan
        self.n_d__t = np.nan
        self.p_d__he3 = np.nan
        self.d_d__he4 = np.nan
        self.p_t__he4 = np.nan
        self.n_he3__he4 = np.nan
        self.p_he3__he4__weak__bet_pos_ = np.nan
        self.d_d__n_he3 = np.nan
        self.d_d__p_t = np.nan
        self.p_t__n_he3 = np.nan
        self.p_t__d_d = np.nan
        self.d_t__n_he4 = np.nan
        self.n_he3__p_t = np.nan
        self.n_he3__d_d = np.nan
        self.d_he3__p_he4 = np.nan
        self.t_he3__d_he4 = np.nan
        self.n_he4__d_t = np.nan
        self.p_he4__d_he3 = np.nan
        self.d_he4__t_he3 = np.nan
        self.p_d__n_p_p = np.nan
        self.t_t__n_n_he4 = np.nan
        self.t_he3__n_p_he4 = np.nan
        self.he3_he3__p_p_he4 = np.nan
        self.n_p_p__p_d = np.nan
        self.n_n_he4__t_t = np.nan
        self.n_p_he4__t_he3 = np.nan
        self.p_p_he4__he3_he3 = np.nan
        self.p__n = np.nan
        self.n__p = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def t__he3__weak__wc12(rate_eval, tf):
    # t --> he3
    rate = 0.0

    # wc12w
    rate += np.exp(  -20.1456)

    rate_eval.t__he3__weak__wc12 = rate

@numba.njit()
def he3__t__weak__electron_capture(rate_eval, tf):
    # he3 --> t
    rate = 0.0

    #   ecw
    rate += np.exp(  -32.462 + -0.21338*tf.T9i + -0.821581*tf.T913i + 11.1241*tf.T913
                  + -0.577338*tf.T9 + 0.0290471*tf.T953 + -0.262705*tf.lnT9)

    rate_eval.he3__t__weak__electron_capture = rate

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
def t__n_d(rate_eval, tf):
    # t --> n + d
    rate = 0.0

    # nk06n
    rate += np.exp(  30.1124 + -72.6136*tf.T9i
                  + 2.5*tf.lnT9)
    # nk06n
    rate += np.exp(  28.869 + -72.6136*tf.T9i
                  + 1.575*tf.lnT9)

    rate_eval.t__n_d = rate

@numba.njit()
def he3__p_d(rate_eval, tf):
    # he3 --> p + d
    rate = 0.0

    # de04 
    rate += np.exp(  32.4383 + -63.7435*tf.T9i + -3.7208*tf.T913i + 0.198654*tf.T913
                  + 1.83333*tf.lnT9)
    # de04n
    rate += np.exp(  31.032 + -63.7435*tf.T9i + -3.7208*tf.T913i + 0.871782*tf.T913
                  + 0.833333*tf.lnT9)

    rate_eval.he3__p_d = rate

@numba.njit()
def he4__n_he3(rate_eval, tf):
    # he4 --> n + he3
    rate = 0.0

    # ka02n
    rate += np.exp(  33.0131 + -238.79*tf.T9i + -1.50147*tf.T913
                  + 2.5*tf.lnT9)
    # ka02n
    rate += np.exp(  29.4845 + -238.79*tf.T9i
                  + 1.5*tf.lnT9)

    rate_eval.he4__n_he3 = rate

@numba.njit()
def he4__p_t(rate_eval, tf):
    # he4 --> p + t
    rate = 0.0

    # cf88n
    rate += np.exp(  33.7327 + -229.932*tf.T9i + -3.869*tf.T913i + 1.45482*tf.T913
                  + 0.577246*tf.T9 + -0.112199*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.he4__p_t = rate

@numba.njit()
def he4__d_d(rate_eval, tf):
    # he4 --> d + d
    rate = 0.0

    # nacrn
    rate += np.exp(  28.2984 + -276.744*tf.T9i + -4.26166*tf.T913i + -0.119233*tf.T913
                  + 0.778829*tf.T9 + -0.0925203*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.he4__d_d = rate

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
def n_d__t(rate_eval, tf):
    # d + n --> t
    rate = 0.0

    # nk06n
    rate += np.exp(  6.60935
                  + 1.0*tf.lnT9)
    # nk06n
    rate += np.exp(  5.36598
                  + 0.075*tf.lnT9)

    rate_eval.n_d__t = rate

@numba.njit()
def p_d__he3(rate_eval, tf):
    # d + p --> he3
    rate = 0.0

    # de04 
    rate += np.exp(  8.93525 + -3.7208*tf.T913i + 0.198654*tf.T913
                  + 0.333333*tf.lnT9)
    # de04n
    rate += np.exp(  7.52898 + -3.7208*tf.T913i + 0.871782*tf.T913
                  + -0.666667*tf.lnT9)

    rate_eval.p_d__he3 = rate

@numba.njit()
def d_d__he4(rate_eval, tf):
    # d + d --> he4
    rate = 0.0

    # nacrn
    rate += np.exp(  3.78177 + -4.26166*tf.T913i + -0.119233*tf.T913
                  + 0.778829*tf.T9 + -0.0925203*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_d__he4 = rate

@numba.njit()
def p_t__he4(rate_eval, tf):
    # t + p --> he4
    rate = 0.0

    # cf88n
    rate += np.exp(  9.76526 + -3.869*tf.T913i + 1.45482*tf.T913
                  + 0.577246*tf.T9 + -0.112199*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_t__he4 = rate

@numba.njit()
def n_he3__he4(rate_eval, tf):
    # he3 + n --> he4
    rate = 0.0

    # ka02n
    rate += np.exp(  9.04572 + -1.50147*tf.T913
                  + 1.0*tf.lnT9)
    # ka02n
    rate += np.exp(  5.51711)

    rate_eval.n_he3__he4 = rate

@numba.njit()
def p_he3__he4__weak__bet_pos_(rate_eval, tf):
    # he3 + p --> he4
    rate = 0.0

    # bet+w
    rate += np.exp(  -27.7611 + -4.30107e-12*tf.T9i + -6.141*tf.T913i + -1.93473e-09*tf.T913
                  + 2.04145e-10*tf.T9 + -1.80372e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_he3__he4__weak__bet_pos_ = rate

@numba.njit()
def d_d__n_he3(rate_eval, tf):
    # d + d --> n + he3
    rate = 0.0

    # gi17n
    rate += np.exp(  19.0876 + -0.00019002*tf.T9i + -4.2292*tf.T913i + 1.6932*tf.T913
                  + -0.0855529*tf.T9 + -1.35709e-25*tf.T953 + -0.734513*tf.lnT9)

    rate_eval.d_d__n_he3 = rate

@numba.njit()
def d_d__p_t(rate_eval, tf):
    # d + d --> p + t
    rate = 0.0

    # go17n
    rate += np.exp(  18.8052 + 4.36209e-05*tf.T9i + -4.32296*tf.T913i + 1.91572*tf.T913
                  + -0.081562*tf.T9 + -3.28804e-22*tf.T953 + -0.879518*tf.lnT9)

    rate_eval.d_d__p_t = rate

@numba.njit()
def p_t__n_he3(rate_eval, tf):
    # t + p --> n + he3
    rate = 0.0

    # de04 
    rate += np.exp(  19.2762 + -8.86352*tf.T9i + 0.0438557*tf.T913
                  + -0.201527*tf.T9 + 0.0153433*tf.T953 + 1.0*tf.lnT9)
    # de04 
    rate += np.exp(  20.3787 + -8.86352*tf.T9i + -0.332788*tf.T913
                  + -0.700485*tf.T9 + 0.0976521*tf.T953)

    rate_eval.p_t__n_he3 = rate

@numba.njit()
def p_t__d_d(rate_eval, tf):
    # t + p --> d + d
    rate = 0.0

    # go17n
    rate += np.exp(  19.3545 + -46.799*tf.T9i + -4.32296*tf.T913i + 1.91572*tf.T913
                  + -0.081562*tf.T9 + -3.28804e-22*tf.T953 + -0.879518*tf.lnT9)

    rate_eval.p_t__d_d = rate

@numba.njit()
def d_t__n_he4(rate_eval, tf):
    # t + d --> n + he4
    rate = 0.0

    # de04 
    rate += np.exp(  39.3457 + -4.5244*tf.T913i + -16.4028*tf.T913
                  + 1.73103*tf.T9 + -0.122966*tf.T953 + 2.31304*tf.lnT9)
    # de04 
    rate += np.exp(  25.1794 + -4.5244*tf.T913i + 0.350337*tf.T913
                  + 0.58747*tf.T9 + -8.84909*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_t__n_he4 = rate

@numba.njit()
def n_he3__p_t(rate_eval, tf):
    # he3 + n --> p + t
    rate = 0.0

    # de04 
    rate += np.exp(  20.3787 + -0.332788*tf.T913
                  + -0.700485*tf.T9 + 0.0976521*tf.T953)
    # de04 
    rate += np.exp(  19.2762 + 0.0438557*tf.T913
                  + -0.201527*tf.T9 + 0.0153433*tf.T953 + 1.0*tf.lnT9)

    rate_eval.n_he3__p_t = rate

@numba.njit()
def n_he3__d_d(rate_eval, tf):
    # he3 + n --> d + d
    rate = 0.0

    # gi17n
    rate += np.exp(  19.6369 + -37.9358*tf.T9i + -4.2292*tf.T913i + 1.6932*tf.T913
                  + -0.0855529*tf.T9 + -1.35709e-25*tf.T953 + -0.734513*tf.lnT9)

    rate_eval.n_he3__d_d = rate

@numba.njit()
def d_he3__p_he4(rate_eval, tf):
    # he3 + d --> p + he4
    rate = 0.0

    # de04 
    rate += np.exp(  24.6839 + -7.182*tf.T913i + 0.473288*tf.T913
                  + 1.46847*tf.T9 + -27.9603*tf.T953 + -0.666667*tf.lnT9)
    # de04 
    rate += np.exp(  41.2969 + -7.182*tf.T913i + -17.1349*tf.T913
                  + 1.36908*tf.T9 + -0.0814423*tf.T953 + 3.35395*tf.lnT9)

    rate_eval.d_he3__p_he4 = rate

@numba.njit()
def t_he3__d_he4(rate_eval, tf):
    # he3 + t --> d + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  22.4207 + -7.733*tf.T913i + -0.133473*tf.T913
                  + -0.294412*tf.T9 + 0.0310968*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_he3__d_he4 = rate

@numba.njit()
def n_he4__d_t(rate_eval, tf):
    # n + he4 --> d + t
    rate = 0.0

    # de04 
    rate += np.exp(  26.8862 + -204.112*tf.T9i + -4.5244*tf.T913i + 0.350337*tf.T913
                  + 0.58747*tf.T9 + -8.84909*tf.T953 + -0.666667*tf.lnT9)
    # de04 
    rate += np.exp(  41.0525 + -204.112*tf.T9i + -4.5244*tf.T913i + -16.4028*tf.T913
                  + 1.73103*tf.T9 + -0.122966*tf.T953 + 2.31304*tf.lnT9)

    rate_eval.n_he4__d_t = rate

@numba.njit()
def p_he4__d_he3(rate_eval, tf):
    # p + he4 --> d + he3
    rate = 0.0

    # de04 
    rate += np.exp(  43.0037 + -212.977*tf.T9i + -7.182*tf.T913i + -17.1349*tf.T913
                  + 1.36908*tf.T9 + -0.0814423*tf.T953 + 3.35395*tf.lnT9)
    # de04 
    rate += np.exp(  26.3907 + -212.977*tf.T9i + -7.182*tf.T913i + 0.473288*tf.T913
                  + 1.46847*tf.T9 + -27.9603*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_he4__d_he3 = rate

@numba.njit()
def d_he4__t_he3(rate_eval, tf):
    # d + he4 --> t + he3
    rate = 0.0

    # cf88n
    rate += np.exp(  22.8851 + -166.176*tf.T9i + -7.733*tf.T913i + -0.133473*tf.T913
                  + -0.294412*tf.T9 + 0.0310968*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_he4__t_he3 = rate

@numba.njit()
def p_d__n_p_p(rate_eval, tf):
    # d + p --> n + p + p
    rate = 0.0

    # cf88n
    rate += np.exp(  17.3271 + -25.82*tf.T9i + -3.72*tf.T913i + 0.946313*tf.T913
                  + 0.105406*tf.T9 + -0.0149431*tf.T953)

    rate_eval.p_d__n_p_p = rate

@numba.njit()
def t_t__n_n_he4(rate_eval, tf):
    # t + t --> n + n + he4
    rate = 0.0

    # cf88r
    rate += np.exp(  21.2361 + -4.872*tf.T913i + -1.72398*tf.T913
                  + 0.684775*tf.T9 + -0.0702582*tf.T953 + 0.333333*tf.lnT9)
    # cf88n
    rate += np.exp(  21.2361 + -4.872*tf.T913i + -0.0328579*tf.T913
                  + -1.13588*tf.T9 + 0.250064*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_t__n_n_he4 = rate

@numba.njit()
def t_he3__n_p_he4(rate_eval, tf):
    # he3 + t --> n + p + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  22.7658 + -7.733*tf.T913i + -0.118902*tf.T913
                  + -0.267393*tf.T9 + 0.0275387*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_he3__n_p_he4 = rate

@numba.njit()
def he3_he3__p_p_he4(rate_eval, tf):
    # he3 + he3 --> p + p + he4
    rate = 0.0

    # nacrn
    rate += np.exp(  24.7788 + -12.277*tf.T913i + -0.103699*tf.T913
                  + -0.0649967*tf.T9 + 0.0168191*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he3_he3__p_p_he4 = rate

@numba.njit()
def n_p_p__p_d(rate_eval, tf):
    # n + p + p --> p + d
    rate = 0.0

    # cf88n
    rate += np.exp(  -4.24034 + -3.72*tf.T913i + 0.946313*tf.T913
                  + 0.105406*tf.T9 + -0.0149431*tf.T953 + -1.5*tf.lnT9)

    rate_eval.n_p_p__p_d = rate

@numba.njit()
def n_n_he4__t_t(rate_eval, tf):
    # n + n + he4 --> t + t
    rate = 0.0

    # cf88r
    rate += np.exp(  -0.560128 + -131.502*tf.T9i + -4.872*tf.T913i + -1.72398*tf.T913
                  + 0.684775*tf.T9 + -0.0702582*tf.T953 + -1.16667*tf.lnT9)
    # cf88n
    rate += np.exp(  -0.560128 + -131.502*tf.T9i + -4.872*tf.T913i + -0.0328579*tf.T913
                  + -1.13588*tf.T9 + 0.250064*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.n_n_he4__t_t = rate

@numba.njit()
def n_p_he4__t_he3(rate_eval, tf):
    # n + p + he4 --> t + he3
    rate = 0.0

    # cf88n
    rate += np.exp(  0.969572 + -140.368*tf.T9i + -7.733*tf.T913i + -0.118902*tf.T913
                  + -0.267393*tf.T9 + 0.0275387*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.n_p_he4__t_he3 = rate

@numba.njit()
def p_p_he4__he3_he3(rate_eval, tf):
    # p + p + he4 --> he3 + he3
    rate = 0.0

    # nacrn
    rate += np.exp(  2.98257 + -149.222*tf.T9i + -12.277*tf.T913i + -0.103699*tf.T913
                  + -0.0649967*tf.T9 + 0.0168191*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.p_p_he4__he3_he3 = rate


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
    t__he3__weak__wc12(rate_eval, tf)
    he3__t__weak__electron_capture(rate_eval, tf)
    d__n_p(rate_eval, tf)
    t__n_d(rate_eval, tf)
    he3__p_d(rate_eval, tf)
    he4__n_he3(rate_eval, tf)
    he4__p_t(rate_eval, tf)
    he4__d_d(rate_eval, tf)
    n_p__d(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    n_d__t(rate_eval, tf)
    p_d__he3(rate_eval, tf)
    d_d__he4(rate_eval, tf)
    p_t__he4(rate_eval, tf)
    n_he3__he4(rate_eval, tf)
    p_he3__he4__weak__bet_pos_(rate_eval, tf)
    d_d__n_he3(rate_eval, tf)
    d_d__p_t(rate_eval, tf)
    p_t__n_he3(rate_eval, tf)
    p_t__d_d(rate_eval, tf)
    d_t__n_he4(rate_eval, tf)
    n_he3__p_t(rate_eval, tf)
    n_he3__d_d(rate_eval, tf)
    d_he3__p_he4(rate_eval, tf)
    t_he3__d_he4(rate_eval, tf)
    n_he4__d_t(rate_eval, tf)
    p_he4__d_he3(rate_eval, tf)
    d_he4__t_he3(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    t_t__n_n_he4(rate_eval, tf)
    t_he3__n_p_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)
    n_n_he4__t_t(rate_eval, tf)
    n_p_he4__t_he3(rate_eval, tf)
    p_p_he4__he3_he3(rate_eval, tf)

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
        rate_eval.p_d__he3 *= scor
        rate_eval.p_d__n_p_p *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d__he4 *= scor
        rate_eval.d_d__n_he3 *= scor
        rate_eval.d_d__p_t *= scor

        scn_fac = ScreenFactors(1, 1, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_t__he4 *= scor
        rate_eval.p_t__n_he3 *= scor
        rate_eval.p_t__d_d *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he3__he4__weak__bet_pos_ *= scor

        scn_fac = ScreenFactors(1, 2, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_t__n_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he3__p_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_he3__d_he4 *= scor
        rate_eval.t_he3__n_p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he4__d_he3 *= scor
        rate_eval.n_p_he4__t_he3 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he4__t_he3 *= scor

        scn_fac = ScreenFactors(1, 3, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_t__n_n_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_he4__he3_he3 *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -2*5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -Y[jn]*rate_eval.n__p
       +Y[jd]*rate_eval.d__n_p
       +Y[jt]*rate_eval.t__n_d
       +Y[jhe4]*rate_eval.he4__n_he3
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +2*5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +Y[jp]*rate_eval.p__n
       )

    dYdt[jp] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -Y[jp]*rate_eval.p__n
       +Y[jd]*rate_eval.d__n_p
       +Y[jhe3]*rate_eval.he3__p_d
       +Y[jhe4]*rate_eval.he4__p_t
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       +Y[jn]*rate_eval.n__p
       )

    dYdt[jd] = (
       -Y[jd]*rate_eval.d__n_p
       -rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       -rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +Y[jt]*rate_eval.t__n_d
       +Y[jhe3]*rate_eval.he3__p_d
       +2*Y[jhe4]*rate_eval.he4__d_d
       +rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       +2*rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       +2*rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       +rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       )

    dYdt[jt] = (
       -Y[jt]*rate_eval.t__he3__weak__wc12
       -Y[jt]*rate_eval.t__n_d
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       -2*5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +rho*ye(Y)*Y[jhe3]*rate_eval.he3__t__weak__electron_capture
       +Y[jhe4]*rate_eval.he4__p_t
       +rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       )

    dYdt[jhe3] = (
       -rho*ye(Y)*Y[jhe3]*rate_eval.he3__t__weak__electron_capture
       -Y[jhe3]*rate_eval.he3__p_d
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       -2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +Y[jt]*rate_eval.t__he3__weak__wc12
       +Y[jhe4]*rate_eval.he4__n_he3
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       )

    dYdt[jhe4] = (
       -Y[jhe4]*rate_eval.he4__n_he3
       -Y[jhe4]*rate_eval.he4__p_t
       -Y[jhe4]*rate_eval.he4__d_d
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       +rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       +5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    t__he3__weak__wc12(rate_eval, tf)
    he3__t__weak__electron_capture(rate_eval, tf)
    d__n_p(rate_eval, tf)
    t__n_d(rate_eval, tf)
    he3__p_d(rate_eval, tf)
    he4__n_he3(rate_eval, tf)
    he4__p_t(rate_eval, tf)
    he4__d_d(rate_eval, tf)
    n_p__d(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    n_d__t(rate_eval, tf)
    p_d__he3(rate_eval, tf)
    d_d__he4(rate_eval, tf)
    p_t__he4(rate_eval, tf)
    n_he3__he4(rate_eval, tf)
    p_he3__he4__weak__bet_pos_(rate_eval, tf)
    d_d__n_he3(rate_eval, tf)
    d_d__p_t(rate_eval, tf)
    p_t__n_he3(rate_eval, tf)
    p_t__d_d(rate_eval, tf)
    d_t__n_he4(rate_eval, tf)
    n_he3__p_t(rate_eval, tf)
    n_he3__d_d(rate_eval, tf)
    d_he3__p_he4(rate_eval, tf)
    t_he3__d_he4(rate_eval, tf)
    n_he4__d_t(rate_eval, tf)
    p_he4__d_he3(rate_eval, tf)
    d_he4__t_he3(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    t_t__n_n_he4(rate_eval, tf)
    t_he3__n_p_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)
    n_n_he4__t_t(rate_eval, tf)
    n_p_he4__t_he3(rate_eval, tf)
    p_p_he4__he3_he3(rate_eval, tf)

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
        rate_eval.p_d__he3 *= scor
        rate_eval.p_d__n_p_p *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d__he4 *= scor
        rate_eval.d_d__n_he3 *= scor
        rate_eval.d_d__p_t *= scor

        scn_fac = ScreenFactors(1, 1, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_t__he4 *= scor
        rate_eval.p_t__n_he3 *= scor
        rate_eval.p_t__d_d *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he3__he4__weak__bet_pos_ *= scor

        scn_fac = ScreenFactors(1, 2, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_t__n_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he3__p_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_he3__d_he4 *= scor
        rate_eval.t_he3__n_p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he4__d_he3 *= scor
        rate_eval.n_p_he4__t_he3 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he4__t_he3 *= scor

        scn_fac = ScreenFactors(1, 3, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_t__n_n_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_he4__he3_he3 *= scor

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -rho*Y[jd]*rate_eval.n_d__t
       -rho*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jhe4]*rate_eval.n_he4__d_t
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       -2*5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -rate_eval.n__p
       )

    jac[jn, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +rho*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jd]*rate_eval.p_d__n_p_p
       +rate_eval.p__n
       )

    jac[jn, jd] = (
       -rho*Y[jn]*rate_eval.n_d__t
       +rate_eval.d__n_p
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__n_he3
       +rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jp]*rate_eval.p_d__n_p_p
       )

    jac[jn, jt] = (
       +rate_eval.t__n_d
       +rho*Y[jp]*rate_eval.p_t__n_he3
       +rho*Y[jd]*rate_eval.d_t__n_he4
       +2*5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       )

    jac[jn, jhe3] = (
       -rho*Y[jn]*rate_eval.n_he3__he4
       -rho*Y[jn]*rate_eval.n_he3__p_t
       -rho*Y[jn]*rate_eval.n_he3__d_d
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       )

    jac[jn, jhe4] = (
       -rho*Y[jn]*rate_eval.n_he4__d_t
       -2*5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       +rate_eval.he4__n_he3
       )

    jac[jp, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +rho*Y[jhe3]*rate_eval.n_he3__p_t
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       +rate_eval.n__p
       )

    jac[jp, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -rate_eval.p__n
       +2*rho*Y[jd]*rate_eval.p_d__n_p_p
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       )

    jac[jp, jd] = (
       -rho*Y[jp]*rate_eval.p_d__he3
       -rho*Y[jp]*rate_eval.p_d__n_p_p
       +rate_eval.d__n_p
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*rho*Y[jp]*rate_eval.p_d__n_p_p
       )

    jac[jp, jt] = (
       -rho*Y[jp]*rate_eval.p_t__he4
       -rho*Y[jp]*rate_eval.p_t__n_he3
       -rho*Y[jp]*rate_eval.p_t__d_d
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       )

    jac[jp, jhe3] = (
       -rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rate_eval.he3__p_d
       +rho*Y[jn]*rate_eval.n_he3__p_t
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       )

    jac[jp, jhe4] = (
       -rho*Y[jp]*rate_eval.p_he4__d_he3
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
       +rate_eval.he4__p_t
       )

    jac[jd, jn] = (
       -rho*Y[jd]*rate_eval.n_d__t
       +rho*Y[jp]*rate_eval.n_p__d
       +2*rho*Y[jhe3]*rate_eval.n_he3__d_d
       +rho*Y[jhe4]*rate_eval.n_he4__d_t
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       )

    jac[jd, jp] = (
       -rho*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jn]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       +2*rho*Y[jt]*rate_eval.p_t__d_d
       +rho*Y[jhe4]*rate_eval.p_he4__d_he3
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       )

    jac[jd, jd] = (
       -rate_eval.d__n_p
       -rho*Y[jn]*rate_eval.n_d__t
       -rho*Y[jp]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__n_he3
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       -rho*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jhe4]*rate_eval.d_he4__t_he3
       -rho*Y[jp]*rate_eval.p_d__n_p_p
       )

    jac[jd, jt] = (
       -rho*Y[jd]*rate_eval.d_t__n_he4
       +rate_eval.t__n_d
       +2*rho*Y[jp]*rate_eval.p_t__d_d
       +rho*Y[jhe3]*rate_eval.t_he3__d_he4
       )

    jac[jd, jhe3] = (
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       +rate_eval.he3__p_d
       +2*rho*Y[jn]*rate_eval.n_he3__d_d
       +rho*Y[jt]*rate_eval.t_he3__d_he4
       )

    jac[jd, jhe4] = (
       -rho*Y[jd]*rate_eval.d_he4__t_he3
       +2*rate_eval.he4__d_d
       +rho*Y[jn]*rate_eval.n_he4__d_t
       +rho*Y[jp]*rate_eval.p_he4__d_he3
       )

    jac[jt, jn] = (
       +rho*Y[jd]*rate_eval.n_d__t
       +rho*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jhe4]*rate_eval.n_he4__d_t
       +2*5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       +rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       )

    jac[jt, jp] = (
       -rho*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jt]*rate_eval.p_t__d_d
       +rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       )

    jac[jt, jd] = (
       -rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jn]*rate_eval.n_d__t
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       +rho*Y[jhe4]*rate_eval.d_he4__t_he3
       )

    jac[jt, jt] = (
       -rate_eval.t__he3__weak__wc12
       -rate_eval.t__n_d
       -rho*Y[jp]*rate_eval.p_t__he4
       -rho*Y[jp]*rate_eval.p_t__n_he3
       -rho*Y[jp]*rate_eval.p_t__d_d
       -rho*Y[jd]*rate_eval.d_t__n_he4
       -rho*Y[jhe3]*rate_eval.t_he3__d_he4
       -2*5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       -rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       )

    jac[jt, jhe3] = (
       -rho*Y[jt]*rate_eval.t_he3__d_he4
       -rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +rho*ye(Y)*rate_eval.he3__t__weak__electron_capture
       +rho*Y[jn]*rate_eval.n_he3__p_t
       )

    jac[jt, jhe4] = (
       +rate_eval.he4__p_t
       +rho*Y[jn]*rate_eval.n_he4__d_t
       +rho*Y[jd]*rate_eval.d_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       +rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       )

    jac[jhe3, jn] = (
       -rho*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jhe3]*rate_eval.n_he3__d_d
       +rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       )

    jac[jhe3, jp] = (
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.p_d__he3
       +rho*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       )

    jac[jhe3, jd] = (
       -rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jp]*rate_eval.p_d__he3
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__n_he3
       +rho*Y[jhe4]*rate_eval.d_he4__t_he3
       )

    jac[jhe3, jt] = (
       -rho*Y[jhe3]*rate_eval.t_he3__d_he4
       -rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +rate_eval.t__he3__weak__wc12
       +rho*Y[jp]*rate_eval.p_t__n_he3
       )

    jac[jhe3, jhe3] = (
       -rho*ye(Y)*rate_eval.he3__t__weak__electron_capture
       -rate_eval.he3__p_d
       -rho*Y[jn]*rate_eval.n_he3__he4
       -rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jn]*rate_eval.n_he3__p_t
       -rho*Y[jn]*rate_eval.n_he3__d_d
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       -rho*Y[jt]*rate_eval.t_he3__d_he4
       -rho*Y[jt]*rate_eval.t_he3__n_p_he4
       -2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       )

    jac[jhe3, jhe4] = (
       +rate_eval.he4__n_he3
       +rho*Y[jp]*rate_eval.p_he4__d_he3
       +rho*Y[jd]*rate_eval.d_he4__t_he3
       +rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
       )

    jac[jhe4, jn] = (
       -rho*Y[jhe4]*rate_eval.n_he4__d_t
       -5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +rho*Y[jhe3]*rate_eval.n_he3__he4
       )

    jac[jhe4, jp] = (
       -rho*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       +rho*Y[jt]*rate_eval.p_t__he4
       +rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       )

    jac[jhe4, jd] = (
       -rho*Y[jhe4]*rate_eval.d_he4__t_he3
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       +rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       )

    jac[jhe4, jt] = (
       +rho*Y[jp]*rate_eval.p_t__he4
       +rho*Y[jd]*rate_eval.d_t__n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__d_he4
       +5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       )

    jac[jhe4, jhe3] = (
       +rho*Y[jn]*rate_eval.n_he3__he4
       +rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +rho*Y[jt]*rate_eval.t_he3__d_he4
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       )

    jac[jhe4, jhe4] = (
       -rate_eval.he4__n_he3
       -rate_eval.he4__p_t
       -rate_eval.he4__d_d
       -rho*Y[jn]*rate_eval.n_he4__d_t
       -rho*Y[jp]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*rate_eval.d_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
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
