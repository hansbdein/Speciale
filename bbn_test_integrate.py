import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import Tfactors, _find_rate_file
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
jd = 2
jt = 3
jhe3 = 4
jhe4 = 5
jli6 = 6
jli7 = 7
jbe7 = 8
nnuc = 9

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jd] = 2
A[jt] = 3
A[jhe3] = 3
A[jhe4] = 4
A[jli6] = 6
A[jli7] = 7
A[jbe7] = 7

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jd] = 1
Z[jt] = 1
Z[jhe3] = 2
Z[jhe4] = 2
Z[jli6] = 3
Z[jli7] = 3
Z[jbe7] = 4

names = []
names.append("n")
names.append("h1")
names.append("h2")
names.append("h3")
names.append("he3")
names.append("he4")
names.append("li6")
names.append("li7")
names.append("be7")

@jitclass([
    ("n__p__weak__wc12", numba.float64),
    ("t__he3__weak__wc12", numba.float64),
    ("he3__t__weak__electron_capture", numba.float64),
    ("be7__li7__weak__electron_capture", numba.float64),
    ("d__n_p", numba.float64),
    ("t__n_d", numba.float64),
    ("he3__p_d", numba.float64),
    ("he4__n_he3", numba.float64),
    ("he4__p_t", numba.float64),
    ("he4__d_d", numba.float64),
    ("li6__he4_d", numba.float64),
    ("li7__n_li6", numba.float64),
    ("li7__he4_t", numba.float64),
    ("be7__p_li6", numba.float64),
    ("be7__he4_he3", numba.float64),
    ("li6__n_p_he4", numba.float64),
    ("n_p__d", numba.float64),
    ("p_p__d__weak__bet_pos_", numba.float64),
    ("p_p__d__weak__electron_capture", numba.float64),
    ("n_d__t", numba.float64),
    ("p_d__he3", numba.float64),
    ("d_d__he4", numba.float64),
    ("he4_d__li6", numba.float64),
    ("p_t__he4", numba.float64),
    ("he4_t__li7", numba.float64),
    ("n_he3__he4", numba.float64),
    ("p_he3__he4__weak__bet_pos_", numba.float64),
    ("he4_he3__be7", numba.float64),
    ("n_li6__li7", numba.float64),
    ("p_li6__be7", numba.float64),
    ("d_d__n_he3", numba.float64),
    ("d_d__p_t", numba.float64),
    ("p_t__n_he3", numba.float64),
    ("p_t__d_d", numba.float64),
    ("d_t__n_he4", numba.float64),
    ("he4_t__n_li6", numba.float64),
    ("n_he3__p_t", numba.float64),
    ("n_he3__d_d", numba.float64),
    ("d_he3__p_he4", numba.float64),
    ("t_he3__d_he4", numba.float64),
    ("he4_he3__p_li6", numba.float64),
    ("n_he4__d_t", numba.float64),
    ("p_he4__d_he3", numba.float64),
    ("d_he4__t_he3", numba.float64),
    ("he4_he4__n_be7", numba.float64),
    ("he4_he4__p_li7", numba.float64),
    ("n_li6__he4_t", numba.float64),
    ("p_li6__he4_he3", numba.float64),
    ("d_li6__n_be7", numba.float64),
    ("d_li6__p_li7", numba.float64),
    ("p_li7__n_be7", numba.float64),
    ("p_li7__d_li6", numba.float64),
    ("p_li7__he4_he4", numba.float64),
    ("n_be7__p_li7", numba.float64),
    ("n_be7__d_li6", numba.float64),
    ("n_be7__he4_he4", numba.float64),
    ("p_d__n_p_p", numba.float64),
    ("t_t__n_n_he4", numba.float64),
    ("t_he3__n_p_he4", numba.float64),
    ("he3_he3__p_p_he4", numba.float64),
    ("d_li7__n_he4_he4", numba.float64),
    ("d_be7__p_he4_he4", numba.float64),
    ("t_li7__n_n_he4_he4", numba.float64),
    ("he3_li7__n_p_he4_he4", numba.float64),
    ("t_be7__n_p_he4_he4", numba.float64),
    ("he3_be7__p_p_he4_he4", numba.float64),
    ("n_p_he4__li6", numba.float64),
    ("n_p_p__p_d", numba.float64),
    ("n_n_he4__t_t", numba.float64),
    ("n_p_he4__t_he3", numba.float64),
    ("p_p_he4__he3_he3", numba.float64),
    ("n_he4_he4__d_li7", numba.float64),
    ("p_he4_he4__d_be7", numba.float64),
    ("n_n_he4_he4__t_li7", numba.float64),
    ("n_p_he4_he4__he3_li7", numba.float64),
    ("n_p_he4_he4__t_be7", numba.float64),
    ("p_p_he4_he4__he3_be7", numba.float64),
])
class RateEval:
    def __init__(self):
        self.n__p__weak__wc12 = np.nan
        self.t__he3__weak__wc12 = np.nan
        self.he3__t__weak__electron_capture = np.nan
        self.be7__li7__weak__electron_capture = np.nan
        self.d__n_p = np.nan
        self.t__n_d = np.nan
        self.he3__p_d = np.nan
        self.he4__n_he3 = np.nan
        self.he4__p_t = np.nan
        self.he4__d_d = np.nan
        self.li6__he4_d = np.nan
        self.li7__n_li6 = np.nan
        self.li7__he4_t = np.nan
        self.be7__p_li6 = np.nan
        self.be7__he4_he3 = np.nan
        self.li6__n_p_he4 = np.nan
        self.n_p__d = np.nan
        self.p_p__d__weak__bet_pos_ = np.nan
        self.p_p__d__weak__electron_capture = np.nan
        self.n_d__t = np.nan
        self.p_d__he3 = np.nan
        self.d_d__he4 = np.nan
        self.he4_d__li6 = np.nan
        self.p_t__he4 = np.nan
        self.he4_t__li7 = np.nan
        self.n_he3__he4 = np.nan
        self.p_he3__he4__weak__bet_pos_ = np.nan
        self.he4_he3__be7 = np.nan
        self.n_li6__li7 = np.nan
        self.p_li6__be7 = np.nan
        self.d_d__n_he3 = np.nan
        self.d_d__p_t = np.nan
        self.p_t__n_he3 = np.nan
        self.p_t__d_d = np.nan
        self.d_t__n_he4 = np.nan
        self.he4_t__n_li6 = np.nan
        self.n_he3__p_t = np.nan
        self.n_he3__d_d = np.nan
        self.d_he3__p_he4 = np.nan
        self.t_he3__d_he4 = np.nan
        self.he4_he3__p_li6 = np.nan
        self.n_he4__d_t = np.nan
        self.p_he4__d_he3 = np.nan
        self.d_he4__t_he3 = np.nan
        self.he4_he4__n_be7 = np.nan
        self.he4_he4__p_li7 = np.nan
        self.n_li6__he4_t = np.nan
        self.p_li6__he4_he3 = np.nan
        self.d_li6__n_be7 = np.nan
        self.d_li6__p_li7 = np.nan
        self.p_li7__n_be7 = np.nan
        self.p_li7__d_li6 = np.nan
        self.p_li7__he4_he4 = np.nan
        self.n_be7__p_li7 = np.nan
        self.n_be7__d_li6 = np.nan
        self.n_be7__he4_he4 = np.nan
        self.p_d__n_p_p = np.nan
        self.t_t__n_n_he4 = np.nan
        self.t_he3__n_p_he4 = np.nan
        self.he3_he3__p_p_he4 = np.nan
        self.d_li7__n_he4_he4 = np.nan
        self.d_be7__p_he4_he4 = np.nan
        self.t_li7__n_n_he4_he4 = np.nan
        self.he3_li7__n_p_he4_he4 = np.nan
        self.t_be7__n_p_he4_he4 = np.nan
        self.he3_be7__p_p_he4_he4 = np.nan
        self.n_p_he4__li6 = np.nan
        self.n_p_p__p_d = np.nan
        self.n_n_he4__t_t = np.nan
        self.n_p_he4__t_he3 = np.nan
        self.p_p_he4__he3_he3 = np.nan
        self.n_he4_he4__d_li7 = np.nan
        self.p_he4_he4__d_be7 = np.nan
        self.n_n_he4_he4__t_li7 = np.nan
        self.n_p_he4_he4__he3_li7 = np.nan
        self.n_p_he4_he4__t_be7 = np.nan
        self.p_p_he4_he4__he3_be7 = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def n__p__weak__wc12(rate_eval, tf):
    # n --> p
    rate = 0.0

    # wc12w
    rate += np.exp(  -6.78161)

    

    rate_eval.n__p__weak__wc12 = rate

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
def be7__li7__weak__electron_capture(rate_eval, tf):
    # be7 --> li7
    rate = 0.0

    #   ecw
    rate += np.exp(  -23.8328 + 3.02033*tf.T913
                  + -0.0742132*tf.T9 + -0.00792386*tf.T953 + -0.650113*tf.lnT9)

    rate_eval.be7__li7__weak__electron_capture = rate

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
def li6__he4_d(rate_eval, tf):
    # li6 --> he4 + d
    rate = 0.0

    # tu19r
    rate += np.exp(  27.5672 + -24.9919*tf.T9i)
    # tu19n
    rate += np.exp(  22.7676 + -17.1028*tf.T9i + -7.55198*tf.T913i + 5.77546*tf.T913
                  + -0.487854*tf.T9 + 0.032833*tf.T953 + 0.376948*tf.lnT9)

    rate_eval.li6__he4_d = rate

@numba.njit()
def li7__n_li6(rate_eval, tf):
    # li7 --> n + li6
    rate = 0.0

    # jz10n
    rate += np.exp(  32.2347 + -84.1369*tf.T9i
                  + 1.5*tf.lnT9)

    rate_eval.li7__n_li6 = rate

@numba.njit()
def li7__he4_t(rate_eval, tf):
    # li7 --> he4 + t
    rate = 0.0

    # de04 
    rate += np.exp(  36.7442 + -28.6283*tf.T9i + -8.0805*tf.T913i + -0.217514*tf.T913
                  + -0.114859*tf.T9 + 0.0470043*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.li7__he4_t = rate

@numba.njit()
def be7__p_li6(rate_eval, tf):
    # be7 --> p + li6
    rate = 0.0

    # nacrn
    rate += np.exp(  37.4661 + -65.0548*tf.T9i + -8.4372*tf.T913i + -0.515473*tf.T913
                  + 0.0285578*tf.T9 + 0.00879731*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.be7__p_li6 = rate

@numba.njit()
def be7__he4_he3(rate_eval, tf):
    # be7 --> he4 + he3
    rate = 0.0

    # cd08n
    rate += np.exp(  38.7379 + -18.4059*tf.T9i + -12.8271*tf.T913i + -0.0308225*tf.T913
                  + -0.654685*tf.T9 + 0.0896331*tf.T953 + 0.833333*tf.lnT9)
    # cd08n
    rate += np.exp(  40.8355 + -18.4059*tf.T9i + -12.8271*tf.T913i + -3.8126*tf.T913
                  + 0.0942285*tf.T9 + -0.00301018*tf.T953 + 2.83333*tf.lnT9)

    rate_eval.be7__he4_he3 = rate

@numba.njit()
def li6__n_p_he4(rate_eval, tf):
    # li6 --> n + p + he4
    rate = 0.0

    # cf88r
    rate += np.exp(  33.4196 + -62.2896*tf.T9i + 1.44987*tf.T913i + -1.42759*tf.T913
                  + 0.0454035*tf.T9 + 0.00471161*tf.T953 + 2.0*tf.lnT9)

    rate_eval.li6__n_p_he4 = rate

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
def he4_d__li6(rate_eval, tf):
    # d + he4 --> li6
    rate = 0.0

    # tu19r
    rate += np.exp(  4.12313 + -7.889*tf.T9i
                  + -1.5*tf.lnT9)
    # tu19n
    rate += np.exp(  -0.676485 + 6.3911e-05*tf.T9i + -7.55198*tf.T913i + 5.77546*tf.T913
                  + -0.487854*tf.T9 + 0.032833*tf.T953 + -1.12305*tf.lnT9)

    rate_eval.he4_d__li6 = rate

@numba.njit()
def p_t__he4(rate_eval, tf):
    # t + p --> he4
    rate = 0.0

    # cf88n
    rate += np.exp(  9.76526 + -3.869*tf.T913i + 1.45482*tf.T913
                  + 0.577246*tf.T9 + -0.112199*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_t__he4 = rate

@numba.njit()
def he4_t__li7(rate_eval, tf):
    # t + he4 --> li7
    rate = 0.0

    # de04 
    rate += np.exp(  13.6162 + -8.0805*tf.T913i + -0.217514*tf.T913
                  + -0.114859*tf.T9 + 0.0470043*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_t__li7 = rate

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
def he4_he3__be7(rate_eval, tf):
    # he3 + he4 --> be7
    rate = 0.0

    # cd08n
    rate += np.exp(  17.7075 + -12.8271*tf.T913i + -3.8126*tf.T913
                  + 0.0942285*tf.T9 + -0.00301018*tf.T953 + 1.33333*tf.lnT9)
    # cd08n
    rate += np.exp(  15.6099 + -12.8271*tf.T913i + -0.0308225*tf.T913
                  + -0.654685*tf.T9 + 0.0896331*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_he3__be7 = rate

@numba.njit()
def n_li6__li7(rate_eval, tf):
    # li6 + n --> li7
    rate = 0.0

    # jz10n
    rate += np.exp(  9.04782)

    rate_eval.n_li6__li7 = rate

@numba.njit()
def p_li6__be7(rate_eval, tf):
    # li6 + p --> be7
    rate = 0.0

    # nacrn
    rate += np.exp(  14.2792 + -8.4372*tf.T913i + -0.515473*tf.T913
                  + 0.0285578*tf.T9 + 0.00879731*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_li6__be7 = rate

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
def he4_t__n_li6(rate_eval, tf):
    # t + he4 --> n + li6
    rate = 0.0

    # cf88n
    rate += np.exp(  19.0085 + -55.494*tf.T9i)
    # cf88r
    rate += np.exp(  21.7239 + -57.884*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_t__n_li6 = rate

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
def he4_he3__p_li6(rate_eval, tf):
    # he3 + he4 --> p + li6
    rate = 0.0

    # pt05n
    rate += np.exp(  24.4064 + -46.6405*tf.T9i + -8.39481*tf.T913i + -0.165254*tf.T913
                  + -0.16936*tf.T9 + 0.0533676*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_he3__p_li6 = rate

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
def he4_he4__n_be7(rate_eval, tf):
    # he4 + he4 --> n + be7
    rate = 0.0

    #  wagn
    rate += np.exp(  19.694 + -220.375*tf.T9i + -0.00210045*tf.T913
                  + 0.000176541*tf.T9 + -1.36797e-05*tf.T953 + 1.00083*tf.lnT9)

    rate_eval.he4_he4__n_be7 = rate

@numba.njit()
def he4_he4__p_li7(rate_eval, tf):
    # he4 + he4 --> p + li7
    rate = 0.0

    # de04r
    rate += np.exp(  23.4325 + -227.465*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  21.9764 + -201.312*tf.T9i + -8.4727*tf.T913i + 0.297934*tf.T913
                  + 0.0582335*tf.T9 + -0.00413383*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  15.7864 + -205.79*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  13.4902 + -201.312*tf.T9i + -8.4727*tf.T913i + 0.417943*tf.T913
                  + 5.34565*tf.T9 + -4.8684*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_he4__p_li7 = rate

@numba.njit()
def n_li6__he4_t(rate_eval, tf):
    # li6 + n --> he4 + t
    rate = 0.0

    # cf88r
    rate += np.exp(  21.665 + -2.39128*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  18.9496 + -0.001281*tf.T9i)

    rate_eval.n_li6__he4_t = rate

@numba.njit()
def p_li6__he4_he3(rate_eval, tf):
    # li6 + p --> he4 + he3
    rate = 0.0

    # pt05n
    rate += np.exp(  24.3475 + -8.39481*tf.T913i + -0.165254*tf.T913
                  + -0.16936*tf.T9 + 0.0533676*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_li6__he4_he3 = rate

@numba.njit()
def d_li6__n_be7(rate_eval, tf):
    # li6 + d --> n + be7
    rate = 0.0

    # mafon
    rate += np.exp(  28.0095 + -4.77456e-12*tf.T9i + -10.259*tf.T913i + -2.01559e-09*tf.T913
                  + 1.99542e-10*tf.T9 + -1.65595e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_li6__n_be7 = rate

@numba.njit()
def d_li6__p_li7(rate_eval, tf):
    # li6 + d --> p + li7
    rate = 0.0

    # mafon
    rate += np.exp(  28.0231 + -10.135*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_li6__p_li7 = rate

@numba.njit()
def p_li7__n_be7(rate_eval, tf):
    # li7 + p --> n + be7
    rate = 0.0

    # db18 
    rate += np.exp(  21.7899 + -19.0779*tf.T9i + -0.30254*tf.T913i + -0.3602*tf.T913
                  + 0.17472*tf.T9 + -0.0223*tf.T953 + -0.4581*tf.lnT9)

    rate_eval.p_li7__n_be7 = rate

@numba.njit()
def p_li7__d_li6(rate_eval, tf):
    # li7 + p --> d + li6
    rate = 0.0

    # mafon
    rate += np.exp(  28.9494 + -58.3239*tf.T9i + -10.135*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.p_li7__d_li6 = rate

@numba.njit()
def p_li7__he4_he4(rate_eval, tf):
    # li7 + p --> he4 + he4
    rate = 0.0

    # de04 
    rate += np.exp(  11.9576 + -8.4727*tf.T913i + 0.417943*tf.T913
                  + 5.34565*tf.T9 + -4.8684*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  21.8999 + -26.1527*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  20.4438 + -8.4727*tf.T913i + 0.297934*tf.T913
                  + 0.0582335*tf.T9 + -0.00413383*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  14.2538 + -4.478*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_li7__he4_he4 = rate

@numba.njit()
def n_be7__p_li7(rate_eval, tf):
    # be7 + n --> p + li7
    rate = 0.0

    # db18 
    rate += np.exp(  21.7899 + 0.000728098*tf.T9i + -0.30254*tf.T913i + -0.3602*tf.T913
                  + 0.17472*tf.T9 + -0.0223*tf.T953 + -0.4581*tf.lnT9)

    rate_eval.n_be7__p_li7 = rate

@numba.njit()
def n_be7__d_li6(rate_eval, tf):
    # be7 + n --> d + li6
    rate = 0.0

    # mafon
    rate += np.exp(  28.9358 + -39.2438*tf.T9i + -10.259*tf.T913i + -2.01559e-09*tf.T913
                  + 1.99542e-10*tf.T9 + -1.65595e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_be7__d_li6 = rate

@numba.njit()
def n_be7__he4_he4(rate_eval, tf):
    # be7 + n --> he4 + he4
    rate = 0.0

    #  wagn
    rate += np.exp(  18.1614 + -0.00210045*tf.T913
                  + 0.000176541*tf.T9 + -1.36797e-05*tf.T953 + 1.00083*tf.lnT9)

    rate_eval.n_be7__he4_he4 = rate

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
def d_li7__n_he4_he4(rate_eval, tf):
    # li7 + d --> n + he4 + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  26.4 + -10.259*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_li7__n_he4_he4 = rate

@numba.njit()
def d_be7__p_he4_he4(rate_eval, tf):
    # be7 + d --> p + he4 + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  27.6987 + -12.428*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_be7__p_he4_he4 = rate

@numba.njit()
def t_li7__n_n_he4_he4(rate_eval, tf):
    # li7 + t --> n + n + he4 + he4
    rate = 0.0

    # mafon
    rate += np.exp(  27.5043 + -5.31692e-12*tf.T9i + -11.333*tf.T913i + -2.24192e-09*tf.T913
                  + 2.21773e-10*tf.T9 + -1.83941e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_li7__n_n_he4_he4 = rate

@numba.njit()
def he3_li7__n_p_he4_he4(rate_eval, tf):
    # li7 + he3 --> n + p + he4 + he4
    rate = 0.0

    # mafon
    rate += np.exp(  30.038 + -4.24733e-12*tf.T9i + -17.989*tf.T913i + -1.57523e-09*tf.T913
                  + 1.45934e-10*tf.T9 + -1.15341e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he3_li7__n_p_he4_he4 = rate

@numba.njit()
def t_be7__n_p_he4_he4(rate_eval, tf):
    # be7 + t --> n + p + he4 + he4
    rate = 0.0

    # mafon
    rate += np.exp(  28.6992 + -6.9004e-12*tf.T9i + -13.792*tf.T913i + -2.92021e-09*tf.T913
                  + 2.89378e-10*tf.T9 + -2.40287e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_be7__n_p_he4_he4 = rate

@numba.njit()
def he3_be7__p_p_he4_he4(rate_eval, tf):
    # be7 + he3 --> p + p + he4 + he4
    rate = 0.0

    # mafon
    rate += np.exp(  31.7435 + -5.45213e-12*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he3_be7__p_p_he4_he4 = rate

@numba.njit()
def n_p_he4__li6(rate_eval, tf):
    # n + p + he4 --> li6
    rate = 0.0

    # cf88r
    rate += np.exp(  -12.2851 + -19.353*tf.T9i + 1.44987*tf.T913i + -1.42759*tf.T913
                  + 0.0454035*tf.T9 + 0.00471161*tf.T953 + -1.0*tf.lnT9)

    rate_eval.n_p_he4__li6 = rate

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

@numba.njit()
def n_he4_he4__d_li7(rate_eval, tf):
    # n + he4 + he4 --> d + li7
    rate = 0.0

    # cf88n
    rate += np.exp(  5.67199 + -175.472*tf.T9i + -10.259*tf.T913i
                  + -2.16667*tf.lnT9)

    rate_eval.n_he4_he4__d_li7 = rate

@numba.njit()
def p_he4_he4__d_be7(rate_eval, tf):
    # p + he4 + he4 --> d + be7
    rate = 0.0

    # cf88n
    rate += np.exp(  6.97069 + -194.561*tf.T9i + -12.428*tf.T913i
                  + -2.16667*tf.lnT9)

    rate_eval.p_he4_he4__d_be7 = rate

@numba.njit()
def n_n_he4_he4__t_li7(rate_eval, tf):
    # n + n + he4 + he4 --> t + li7
    rate = 0.0

    # mafon
    rate += np.exp(  -17.4199 + -102.867*tf.T9i + -11.333*tf.T913i + -2.24192e-09*tf.T913
                  + 2.21773e-10*tf.T9 + -1.83941e-11*tf.T953 + -3.66667*tf.lnT9)

    rate_eval.n_n_he4_he4__t_li7 = rate

@numba.njit()
def n_p_he4_he4__he3_li7(rate_eval, tf):
    # n + p + he4 + he4 --> he3 + li7
    rate = 0.0

    # mafon
    rate += np.exp(  -14.8862 + -111.725*tf.T9i + -17.989*tf.T913i + -1.57523e-09*tf.T913
                  + 1.45934e-10*tf.T9 + -1.15341e-11*tf.T953 + -3.66667*tf.lnT9)

    rate_eval.n_p_he4_he4__he3_li7 = rate

@numba.njit()
def n_p_he4_he4__t_be7(rate_eval, tf):
    # n + p + he4 + he4 --> t + be7
    rate = 0.0

    # mafon
    rate += np.exp(  -16.225 + -121.949*tf.T9i + -13.792*tf.T913i + -2.92021e-09*tf.T913
                  + 2.89378e-10*tf.T9 + -2.40287e-11*tf.T953 + -3.66667*tf.lnT9)

    rate_eval.n_p_he4_he4__t_be7 = rate

@numba.njit()
def p_p_he4_he4__he3_be7(rate_eval, tf):
    # p + p + he4 + he4 --> he3 + be7
    rate = 0.0

    # mafon
    rate += np.exp(  -13.1807 + -130.807*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -3.66667*tf.lnT9)

    rate_eval.p_p_he4_he4__he3_be7 = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n__p__weak__wc12(rate_eval, tf)
    t__he3__weak__wc12(rate_eval, tf)
    he3__t__weak__electron_capture(rate_eval, tf)
    be7__li7__weak__electron_capture(rate_eval, tf)
    d__n_p(rate_eval, tf)
    t__n_d(rate_eval, tf)
    he3__p_d(rate_eval, tf)
    he4__n_he3(rate_eval, tf)
    he4__p_t(rate_eval, tf)
    he4__d_d(rate_eval, tf)
    li6__he4_d(rate_eval, tf)
    li7__n_li6(rate_eval, tf)
    li7__he4_t(rate_eval, tf)
    be7__p_li6(rate_eval, tf)
    be7__he4_he3(rate_eval, tf)
    li6__n_p_he4(rate_eval, tf)
    n_p__d(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    n_d__t(rate_eval, tf)
    p_d__he3(rate_eval, tf)
    d_d__he4(rate_eval, tf)
    he4_d__li6(rate_eval, tf)
    p_t__he4(rate_eval, tf)
    he4_t__li7(rate_eval, tf)
    n_he3__he4(rate_eval, tf)
    p_he3__he4__weak__bet_pos_(rate_eval, tf)
    he4_he3__be7(rate_eval, tf)
    n_li6__li7(rate_eval, tf)
    p_li6__be7(rate_eval, tf)
    d_d__n_he3(rate_eval, tf)
    d_d__p_t(rate_eval, tf)
    p_t__n_he3(rate_eval, tf)
    p_t__d_d(rate_eval, tf)
    d_t__n_he4(rate_eval, tf)
    he4_t__n_li6(rate_eval, tf)
    n_he3__p_t(rate_eval, tf)
    n_he3__d_d(rate_eval, tf)
    d_he3__p_he4(rate_eval, tf)
    t_he3__d_he4(rate_eval, tf)
    he4_he3__p_li6(rate_eval, tf)
    n_he4__d_t(rate_eval, tf)
    p_he4__d_he3(rate_eval, tf)
    d_he4__t_he3(rate_eval, tf)
    he4_he4__n_be7(rate_eval, tf)
    he4_he4__p_li7(rate_eval, tf)
    n_li6__he4_t(rate_eval, tf)
    p_li6__he4_he3(rate_eval, tf)
    d_li6__n_be7(rate_eval, tf)
    d_li6__p_li7(rate_eval, tf)
    p_li7__n_be7(rate_eval, tf)
    p_li7__d_li6(rate_eval, tf)
    p_li7__he4_he4(rate_eval, tf)
    n_be7__p_li7(rate_eval, tf)
    n_be7__d_li6(rate_eval, tf)
    n_be7__he4_he4(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    t_t__n_n_he4(rate_eval, tf)
    t_he3__n_p_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    d_li7__n_he4_he4(rate_eval, tf)
    d_be7__p_he4_he4(rate_eval, tf)
    t_li7__n_n_he4_he4(rate_eval, tf)
    he3_li7__n_p_he4_he4(rate_eval, tf)
    t_be7__n_p_he4_he4(rate_eval, tf)
    he3_be7__p_p_he4_he4(rate_eval, tf)
    n_p_he4__li6(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)
    n_n_he4__t_t(rate_eval, tf)
    n_p_he4__t_he3(rate_eval, tf)
    p_p_he4__he3_he3(rate_eval, tf)
    n_he4_he4__d_li7(rate_eval, tf)
    p_he4_he4__d_be7(rate_eval, tf)
    n_n_he4_he4__t_li7(rate_eval, tf)
    n_p_he4_he4__he3_li7(rate_eval, tf)
    n_p_he4_he4__t_be7(rate_eval, tf)
    p_p_he4_he4__he3_be7(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p__d__weak__bet_pos_ *= scor
        rate_eval.p_p__d__weak__electron_capture *= scor
        rate_eval.n_p_p__p_d *= scor
        rate_eval.p_p_he4_he4__he3_be7 *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d__he3 *= scor
        rate_eval.p_d__n_p_p *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d__he4 *= scor
        rate_eval.d_d__n_he3 *= scor
        rate_eval.d_d__p_t *= scor

        scn_fac = ScreenFactors(1, 2, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_d__li6 *= scor
        rate_eval.d_he4__t_he3 *= scor

        scn_fac = ScreenFactors(1, 1, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_t__he4 *= scor
        rate_eval.p_t__n_he3 *= scor
        rate_eval.p_t__d_d *= scor

        scn_fac = ScreenFactors(1, 3, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_t__li7 *= scor
        rate_eval.he4_t__n_li6 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he3__he4__weak__bet_pos_ *= scor

        scn_fac = ScreenFactors(2, 4, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_he3__be7 *= scor
        rate_eval.he4_he3__p_li6 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 6)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li6__be7 *= scor
        rate_eval.p_li6__he4_he3 *= scor

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
        rate_eval.n_p_he4__li6 *= scor
        rate_eval.n_p_he4__t_he3 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_he4__n_be7 *= scor
        rate_eval.he4_he4__p_li7 *= scor
        rate_eval.n_he4_he4__d_li7 *= scor
        rate_eval.n_n_he4_he4__t_li7 *= scor

        scn_fac = ScreenFactors(1, 2, 3, 6)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li6__n_be7 *= scor
        rate_eval.d_li6__p_li7 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li7__n_be7 *= scor
        rate_eval.p_li7__d_li6 *= scor
        rate_eval.p_li7__he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_t__n_n_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li7__n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_be7__p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_li7__n_n_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_li7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_be7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_be7__p_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_he4__he3_he3 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he4_he4__d_be7 *= scor
        rate_eval.n_p_he4_he4__he3_li7 *= scor
        rate_eval.n_p_he4_he4__t_be7 *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -Y[jn]*rate_eval.n__p__weak__wc12
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__he4_he4
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -2*5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       +Y[jd]*rate_eval.d__n_p
       +Y[jt]*rate_eval.t__n_d
       +Y[jhe4]*rate_eval.he4__n_he3
       +Y[jli7]*rate_eval.li7__n_li6
       +Y[jli6]*rate_eval.li6__n_p_he4
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +2*5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    dYdt[jp] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       +Y[jn]*rate_eval.n__p__weak__wc12
       +Y[jd]*rate_eval.d__n_p
       +Y[jhe3]*rate_eval.he3__p_d
       +Y[jhe4]*rate_eval.he4__p_t
       +Y[jbe7]*rate_eval.be7__p_li6
       +Y[jli6]*rate_eval.li6__n_p_he4
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       +2*rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       )

    dYdt[jd] = (
       -Y[jd]*rate_eval.d__n_p
       -rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       -rho*Y[jd]*Y[jhe4]*rate_eval.he4_d__li6
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       -rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +Y[jt]*rate_eval.t__n_d
       +Y[jhe3]*rate_eval.he3__p_d
       +2*Y[jhe4]*rate_eval.he4__d_d
       +Y[jli6]*rate_eval.li6__he4_d
       +rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       +2*rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       +2*rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       +rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       )

    dYdt[jt] = (
       -Y[jt]*rate_eval.t__he3__weak__wc12
       -Y[jt]*rate_eval.t__n_d
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__li7
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       -2*5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       -rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +rho*ye(Y)*Y[jhe3]*rate_eval.he3__t__weak__electron_capture
       +Y[jhe4]*rate_eval.he4__p_t
       +Y[jli7]*rate_eval.li7__he4_t
       +rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       +rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       +2*5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       )

    dYdt[jhe3] = (
       -rho*ye(Y)*Y[jhe3]*rate_eval.he3__t__weak__electron_capture
       -Y[jhe3]*rate_eval.he3__p_d
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       -rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       -2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       -rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       -rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +Y[jt]*rate_eval.t__he3__weak__wc12
       +Y[jhe4]*rate_eval.he4__n_he3
       +Y[jbe7]*rate_eval.be7__he4_he3
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       +rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       +2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       )

    dYdt[jhe4] = (
       -Y[jhe4]*rate_eval.he4__n_he3
       -Y[jhe4]*rate_eval.he4__p_t
       -Y[jhe4]*rate_eval.he4__d_d
       -rho*Y[jd]*Y[jhe4]*rate_eval.he4_d__li6
       -rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__li7
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       -2*5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       -2*5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       +Y[jli6]*rate_eval.li6__he4_d
       +Y[jli7]*rate_eval.li7__he4_t
       +Y[jbe7]*rate_eval.be7__he4_he3
       +Y[jli6]*rate_eval.li6__n_p_he4
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       +rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__d_he4
       +rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       +rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       +2*rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       +2*rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__he4_he4
       +5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +2*rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +2*rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    dYdt[jli6] = (
       -Y[jli6]*rate_eval.li6__he4_d
       -Y[jli6]*rate_eval.li6__n_p_he4
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       +Y[jli7]*rate_eval.li7__n_li6
       +Y[jbe7]*rate_eval.be7__p_li6
       +rho*Y[jd]*Y[jhe4]*rate_eval.he4_d__li6
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       )

    dYdt[jli7] = (
       -Y[jli7]*rate_eval.li7__n_li6
       -Y[jli7]*rate_eval.li7__he4_t
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       -rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__li7
       +rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       +2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       )

    dYdt[jbe7] = (
       -rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       -Y[jbe7]*rate_eval.be7__p_li6
       -Y[jbe7]*rate_eval.be7__he4_he3
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__he4_he4
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       -rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       -rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       +rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       +2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       )

    return dYdt
