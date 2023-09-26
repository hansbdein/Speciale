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
jli6 = 6
jli7 = 7
jli8 = 8
jbe7 = 9
jbe9 = 10
jb8 = 11
jb10 = 12
jb11 = 13
jb12 = 14
jc11 = 15
jc12 = 16
jc13 = 17
jc14 = 18
jn12 = 19
jn13 = 20
jn14 = 21
jn15 = 22
jo14 = 23
jo15 = 24
jo16 = 25
nnuc = 26

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jd] = 2
A[jt] = 3
A[jhe3] = 3
A[jhe4] = 4
A[jli6] = 6
A[jli7] = 7
A[jli8] = 8
A[jbe7] = 7
A[jbe9] = 9
A[jb8] = 8
A[jb10] = 10
A[jb11] = 11
A[jb12] = 12
A[jc11] = 11
A[jc12] = 12
A[jc13] = 13
A[jc14] = 14
A[jn12] = 12
A[jn13] = 13
A[jn14] = 14
A[jn15] = 15
A[jo14] = 14
A[jo15] = 15
A[jo16] = 16

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jd] = 1
Z[jt] = 1
Z[jhe3] = 2
Z[jhe4] = 2
Z[jli6] = 3
Z[jli7] = 3
Z[jli8] = 3
Z[jbe7] = 4
Z[jbe9] = 4
Z[jb8] = 5
Z[jb10] = 5
Z[jb11] = 5
Z[jb12] = 5
Z[jc11] = 6
Z[jc12] = 6
Z[jc13] = 6
Z[jc14] = 6
Z[jn12] = 7
Z[jn13] = 7
Z[jn14] = 7
Z[jn15] = 7
Z[jo14] = 8
Z[jo15] = 8
Z[jo16] = 8

names = []
names.append("n")
names.append("h1")
names.append("h2")
names.append("h3")
names.append("he3")
names.append("he4")
names.append("li6")
names.append("li7")
names.append("li8")
names.append("be7")
names.append("be9")
names.append("b8")
names.append("b10")
names.append("b11")
names.append("b12")
names.append("c11")
names.append("c12")
names.append("c13")
names.append("c14")
names.append("n12")
names.append("n13")
names.append("n14")
names.append("n15")
names.append("o14")
names.append("o15")
names.append("o16")

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
    ("be7__li7__weak__electron_capture", numba.float64),
    ("b12__c12__weak__wc17", numba.float64),
    ("c11__b11__weak__wc12", numba.float64),
    ("c14__n14__weak__wc12", numba.float64),
    ("n12__c12__weak__wc12", numba.float64),
    ("n13__c13__weak__wc12", numba.float64),
    ("o14__n14__weak__wc12", numba.float64),
    ("o15__n15__weak__wc12", numba.float64),
    ("d__n_p", numba.float64),
    ("t__n_d", numba.float64),
    ("he3__p_d", numba.float64),
    ("he4__n_he3", numba.float64),
    ("he4__p_t", numba.float64),
    ("he4__d_d", numba.float64),
    ("li6__he4_d", numba.float64),
    ("li7__n_li6", numba.float64),
    ("li7__he4_t", numba.float64),
    ("li8__n_li7", numba.float64),
    ("li8__he4_he4__weak__wc12", numba.float64),
    ("be7__p_li6", numba.float64),
    ("be7__he4_he3", numba.float64),
    ("b8__p_be7", numba.float64),
    ("b8__he4_he4__weak__wc12", numba.float64),
    ("b10__p_be9", numba.float64),
    ("b10__he4_li6", numba.float64),
    ("b11__n_b10", numba.float64),
    ("b11__he4_li7", numba.float64),
    ("b12__n_b11", numba.float64),
    ("b12__he4_li8", numba.float64),
    ("c11__p_b10", numba.float64),
    ("c11__he4_be7", numba.float64),
    ("c12__n_c11", numba.float64),
    ("c12__p_b11", numba.float64),
    ("c13__n_c12", numba.float64),
    ("c14__n_c13", numba.float64),
    ("n12__p_c11", numba.float64),
    ("n13__p_c12", numba.float64),
    ("n14__n_n13", numba.float64),
    ("n14__p_c13", numba.float64),
    ("n15__n_n14", numba.float64),
    ("n15__p_c14", numba.float64),
    ("o14__p_n13", numba.float64),
    ("o15__n_o14", numba.float64),
    ("o15__p_n14", numba.float64),
    ("o16__n_o15", numba.float64),
    ("o16__p_n15", numba.float64),
    ("o16__he4_c12", numba.float64),
    ("li6__n_p_he4", numba.float64),
    ("be9__n_he4_he4", numba.float64),
    ("c12__he4_he4_he4", numba.float64),
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
    ("he4_li6__b10", numba.float64),
    ("n_li7__li8", numba.float64),
    ("he4_li7__b11", numba.float64),
    ("he4_li8__b12", numba.float64),
    ("p_be7__b8", numba.float64),
    ("he4_be7__c11", numba.float64),
    ("p_be9__b10", numba.float64),
    ("n_b10__b11", numba.float64),
    ("p_b10__c11", numba.float64),
    ("n_b11__b12", numba.float64),
    ("p_b11__c12", numba.float64),
    ("n_c11__c12", numba.float64),
    ("p_c11__n12", numba.float64),
    ("n_c12__c13", numba.float64),
    ("p_c12__n13", numba.float64),
    ("he4_c12__o16", numba.float64),
    ("n_c13__c14", numba.float64),
    ("p_c13__n14", numba.float64),
    ("p_c14__n15", numba.float64),
    ("n_n13__n14", numba.float64),
    ("p_n13__o14", numba.float64),
    ("n_n14__n15", numba.float64),
    ("p_n14__o15", numba.float64),
    ("p_n15__o16", numba.float64),
    ("n_o14__o15", numba.float64),
    ("n_o15__o16", numba.float64),
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
    ("he4_li6__p_be9", numba.float64),
    ("p_li7__n_be7", numba.float64),
    ("p_li7__d_li6", numba.float64),
    ("p_li7__he4_he4", numba.float64),
    ("d_li7__p_li8", numba.float64),
    ("t_li7__n_be9", numba.float64),
    ("t_li7__d_li8", numba.float64),
    ("he4_li7__n_b10", numba.float64),
    ("p_li8__d_li7", numba.float64),
    ("d_li8__n_be9", numba.float64),
    ("d_li8__t_li7", numba.float64),
    ("he4_li8__n_b11", numba.float64),
    ("n_be7__p_li7", numba.float64),
    ("n_be7__d_li6", numba.float64),
    ("n_be7__he4_he4", numba.float64),
    ("he4_be7__p_b10", numba.float64),
    ("n_be9__d_li8", numba.float64),
    ("n_be9__t_li7", numba.float64),
    ("p_be9__he4_li6", numba.float64),
    ("t_be9__n_b11", numba.float64),
    ("he4_be9__n_c12", numba.float64),
    ("he4_be9__p_b12", numba.float64),
    ("he4_b8__p_c11", numba.float64),
    ("n_b10__he4_li7", numba.float64),
    ("p_b10__he4_be7", numba.float64),
    ("he4_b10__n_n13", numba.float64),
    ("he4_b10__p_c13", numba.float64),
    ("n_b11__t_be9", numba.float64),
    ("n_b11__he4_li8", numba.float64),
    ("p_b11__n_c11", numba.float64),
    ("he4_b11__n_n14", numba.float64),
    ("he4_b11__p_c14", numba.float64),
    ("p_b12__n_c12", numba.float64),
    ("p_b12__he4_be9", numba.float64),
    ("he4_b12__n_n15", numba.float64),
    ("n_c11__p_b11", numba.float64),
    ("p_c11__he4_b8", numba.float64),
    ("he4_c11__n_o14", numba.float64),
    ("he4_c11__p_n14", numba.float64),
    ("n_c12__p_b12", numba.float64),
    ("n_c12__he4_be9", numba.float64),
    ("he4_c12__n_o15", numba.float64),
    ("he4_c12__p_n15", numba.float64),
    ("p_c13__n_n13", numba.float64),
    ("p_c13__he4_b10", numba.float64),
    ("d_c13__n_n14", numba.float64),
    ("he4_c13__n_o16", numba.float64),
    ("p_c14__n_n14", numba.float64),
    ("p_c14__he4_b11", numba.float64),
    ("d_c14__n_n15", numba.float64),
    ("he4_n12__p_o15", numba.float64),
    ("n_n13__p_c13", numba.float64),
    ("n_n13__he4_b10", numba.float64),
    ("he4_n13__p_o16", numba.float64),
    ("n_n14__p_c14", numba.float64),
    ("n_n14__d_c13", numba.float64),
    ("n_n14__he4_b11", numba.float64),
    ("p_n14__n_o14", numba.float64),
    ("p_n14__he4_c11", numba.float64),
    ("n_n15__d_c14", numba.float64),
    ("n_n15__he4_b12", numba.float64),
    ("p_n15__n_o15", numba.float64),
    ("p_n15__he4_c12", numba.float64),
    ("n_o14__p_n14", numba.float64),
    ("n_o14__he4_c11", numba.float64),
    ("n_o15__p_n15", numba.float64),
    ("n_o15__he4_c12", numba.float64),
    ("p_o15__he4_n12", numba.float64),
    ("n_o16__he4_c13", numba.float64),
    ("p_o16__he4_n13", numba.float64),
    ("p_d__n_p_p", numba.float64),
    ("t_t__n_n_he4", numba.float64),
    ("t_he3__n_p_he4", numba.float64),
    ("he3_he3__p_p_he4", numba.float64),
    ("d_li7__n_he4_he4", numba.float64),
    ("p_li8__n_he4_he4", numba.float64),
    ("d_be7__p_he4_he4", numba.float64),
    ("p_be9__d_he4_he4", numba.float64),
    ("n_b8__p_he4_he4", numba.float64),
    ("p_b11__he4_he4_he4", numba.float64),
    ("n_c11__he4_he4_he4", numba.float64),
    ("t_li7__n_n_he4_he4", numba.float64),
    ("he3_li7__n_p_he4_he4", numba.float64),
    ("t_be7__n_p_he4_he4", numba.float64),
    ("he3_be7__p_p_he4_he4", numba.float64),
    ("p_be9__n_p_he4_he4", numba.float64),
    ("n_p_he4__li6", numba.float64),
    ("n_he4_he4__be9", numba.float64),
    ("he4_he4_he4__c12", numba.float64),
    ("n_p_p__p_d", numba.float64),
    ("n_n_he4__t_t", numba.float64),
    ("n_p_he4__t_he3", numba.float64),
    ("p_p_he4__he3_he3", numba.float64),
    ("n_he4_he4__p_li8", numba.float64),
    ("n_he4_he4__d_li7", numba.float64),
    ("p_he4_he4__n_b8", numba.float64),
    ("p_he4_he4__d_be7", numba.float64),
    ("d_he4_he4__p_be9", numba.float64),
    ("he4_he4_he4__n_c11", numba.float64),
    ("he4_he4_he4__p_b11", numba.float64),
    ("n_n_he4_he4__t_li7", numba.float64),
    ("n_p_he4_he4__he3_li7", numba.float64),
    ("n_p_he4_he4__t_be7", numba.float64),
    ("n_p_he4_he4__p_be9", numba.float64),
    ("p_p_he4_he4__he3_be7", numba.float64),
    ("p__n", numba.float64),
    ("n__p", numba.float64),
])
class RateEval:
    def __init__(self):
        self.t__he3__weak__wc12 = np.nan
        self.he3__t__weak__electron_capture = np.nan
        self.be7__li7__weak__electron_capture = np.nan
        self.b12__c12__weak__wc17 = np.nan
        self.c11__b11__weak__wc12 = np.nan
        self.c14__n14__weak__wc12 = np.nan
        self.n12__c12__weak__wc12 = np.nan
        self.n13__c13__weak__wc12 = np.nan
        self.o14__n14__weak__wc12 = np.nan
        self.o15__n15__weak__wc12 = np.nan
        self.d__n_p = np.nan
        self.t__n_d = np.nan
        self.he3__p_d = np.nan
        self.he4__n_he3 = np.nan
        self.he4__p_t = np.nan
        self.he4__d_d = np.nan
        self.li6__he4_d = np.nan
        self.li7__n_li6 = np.nan
        self.li7__he4_t = np.nan
        self.li8__n_li7 = np.nan
        self.li8__he4_he4__weak__wc12 = np.nan
        self.be7__p_li6 = np.nan
        self.be7__he4_he3 = np.nan
        self.b8__p_be7 = np.nan
        self.b8__he4_he4__weak__wc12 = np.nan
        self.b10__p_be9 = np.nan
        self.b10__he4_li6 = np.nan
        self.b11__n_b10 = np.nan
        self.b11__he4_li7 = np.nan
        self.b12__n_b11 = np.nan
        self.b12__he4_li8 = np.nan
        self.c11__p_b10 = np.nan
        self.c11__he4_be7 = np.nan
        self.c12__n_c11 = np.nan
        self.c12__p_b11 = np.nan
        self.c13__n_c12 = np.nan
        self.c14__n_c13 = np.nan
        self.n12__p_c11 = np.nan
        self.n13__p_c12 = np.nan
        self.n14__n_n13 = np.nan
        self.n14__p_c13 = np.nan
        self.n15__n_n14 = np.nan
        self.n15__p_c14 = np.nan
        self.o14__p_n13 = np.nan
        self.o15__n_o14 = np.nan
        self.o15__p_n14 = np.nan
        self.o16__n_o15 = np.nan
        self.o16__p_n15 = np.nan
        self.o16__he4_c12 = np.nan
        self.li6__n_p_he4 = np.nan
        self.be9__n_he4_he4 = np.nan
        self.c12__he4_he4_he4 = np.nan
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
        self.he4_li6__b10 = np.nan
        self.n_li7__li8 = np.nan
        self.he4_li7__b11 = np.nan
        self.he4_li8__b12 = np.nan
        self.p_be7__b8 = np.nan
        self.he4_be7__c11 = np.nan
        self.p_be9__b10 = np.nan
        self.n_b10__b11 = np.nan
        self.p_b10__c11 = np.nan
        self.n_b11__b12 = np.nan
        self.p_b11__c12 = np.nan
        self.n_c11__c12 = np.nan
        self.p_c11__n12 = np.nan
        self.n_c12__c13 = np.nan
        self.p_c12__n13 = np.nan
        self.he4_c12__o16 = np.nan
        self.n_c13__c14 = np.nan
        self.p_c13__n14 = np.nan
        self.p_c14__n15 = np.nan
        self.n_n13__n14 = np.nan
        self.p_n13__o14 = np.nan
        self.n_n14__n15 = np.nan
        self.p_n14__o15 = np.nan
        self.p_n15__o16 = np.nan
        self.n_o14__o15 = np.nan
        self.n_o15__o16 = np.nan
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
        self.he4_li6__p_be9 = np.nan
        self.p_li7__n_be7 = np.nan
        self.p_li7__d_li6 = np.nan
        self.p_li7__he4_he4 = np.nan
        self.d_li7__p_li8 = np.nan
        self.t_li7__n_be9 = np.nan
        self.t_li7__d_li8 = np.nan
        self.he4_li7__n_b10 = np.nan
        self.p_li8__d_li7 = np.nan
        self.d_li8__n_be9 = np.nan
        self.d_li8__t_li7 = np.nan
        self.he4_li8__n_b11 = np.nan
        self.n_be7__p_li7 = np.nan
        self.n_be7__d_li6 = np.nan
        self.n_be7__he4_he4 = np.nan
        self.he4_be7__p_b10 = np.nan
        self.n_be9__d_li8 = np.nan
        self.n_be9__t_li7 = np.nan
        self.p_be9__he4_li6 = np.nan
        self.t_be9__n_b11 = np.nan
        self.he4_be9__n_c12 = np.nan
        self.he4_be9__p_b12 = np.nan
        self.he4_b8__p_c11 = np.nan
        self.n_b10__he4_li7 = np.nan
        self.p_b10__he4_be7 = np.nan
        self.he4_b10__n_n13 = np.nan
        self.he4_b10__p_c13 = np.nan
        self.n_b11__t_be9 = np.nan
        self.n_b11__he4_li8 = np.nan
        self.p_b11__n_c11 = np.nan
        self.he4_b11__n_n14 = np.nan
        self.he4_b11__p_c14 = np.nan
        self.p_b12__n_c12 = np.nan
        self.p_b12__he4_be9 = np.nan
        self.he4_b12__n_n15 = np.nan
        self.n_c11__p_b11 = np.nan
        self.p_c11__he4_b8 = np.nan
        self.he4_c11__n_o14 = np.nan
        self.he4_c11__p_n14 = np.nan
        self.n_c12__p_b12 = np.nan
        self.n_c12__he4_be9 = np.nan
        self.he4_c12__n_o15 = np.nan
        self.he4_c12__p_n15 = np.nan
        self.p_c13__n_n13 = np.nan
        self.p_c13__he4_b10 = np.nan
        self.d_c13__n_n14 = np.nan
        self.he4_c13__n_o16 = np.nan
        self.p_c14__n_n14 = np.nan
        self.p_c14__he4_b11 = np.nan
        self.d_c14__n_n15 = np.nan
        self.he4_n12__p_o15 = np.nan
        self.n_n13__p_c13 = np.nan
        self.n_n13__he4_b10 = np.nan
        self.he4_n13__p_o16 = np.nan
        self.n_n14__p_c14 = np.nan
        self.n_n14__d_c13 = np.nan
        self.n_n14__he4_b11 = np.nan
        self.p_n14__n_o14 = np.nan
        self.p_n14__he4_c11 = np.nan
        self.n_n15__d_c14 = np.nan
        self.n_n15__he4_b12 = np.nan
        self.p_n15__n_o15 = np.nan
        self.p_n15__he4_c12 = np.nan
        self.n_o14__p_n14 = np.nan
        self.n_o14__he4_c11 = np.nan
        self.n_o15__p_n15 = np.nan
        self.n_o15__he4_c12 = np.nan
        self.p_o15__he4_n12 = np.nan
        self.n_o16__he4_c13 = np.nan
        self.p_o16__he4_n13 = np.nan
        self.p_d__n_p_p = np.nan
        self.t_t__n_n_he4 = np.nan
        self.t_he3__n_p_he4 = np.nan
        self.he3_he3__p_p_he4 = np.nan
        self.d_li7__n_he4_he4 = np.nan
        self.p_li8__n_he4_he4 = np.nan
        self.d_be7__p_he4_he4 = np.nan
        self.p_be9__d_he4_he4 = np.nan
        self.n_b8__p_he4_he4 = np.nan
        self.p_b11__he4_he4_he4 = np.nan
        self.n_c11__he4_he4_he4 = np.nan
        self.t_li7__n_n_he4_he4 = np.nan
        self.he3_li7__n_p_he4_he4 = np.nan
        self.t_be7__n_p_he4_he4 = np.nan
        self.he3_be7__p_p_he4_he4 = np.nan
        self.p_be9__n_p_he4_he4 = np.nan
        self.n_p_he4__li6 = np.nan
        self.n_he4_he4__be9 = np.nan
        self.he4_he4_he4__c12 = np.nan
        self.n_p_p__p_d = np.nan
        self.n_n_he4__t_t = np.nan
        self.n_p_he4__t_he3 = np.nan
        self.p_p_he4__he3_he3 = np.nan
        self.n_he4_he4__p_li8 = np.nan
        self.n_he4_he4__d_li7 = np.nan
        self.p_he4_he4__n_b8 = np.nan
        self.p_he4_he4__d_be7 = np.nan
        self.d_he4_he4__p_be9 = np.nan
        self.he4_he4_he4__n_c11 = np.nan
        self.he4_he4_he4__p_b11 = np.nan
        self.n_n_he4_he4__t_li7 = np.nan
        self.n_p_he4_he4__he3_li7 = np.nan
        self.n_p_he4_he4__t_be7 = np.nan
        self.n_p_he4_he4__p_be9 = np.nan
        self.p_p_he4_he4__he3_be7 = np.nan
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
def be7__li7__weak__electron_capture(rate_eval, tf):
    # be7 --> li7
    rate = 0.0

    #   ecw
    rate += np.exp(  -23.8328 + 3.02033*tf.T913
                  + -0.0742132*tf.T9 + -0.00792386*tf.T953 + -0.650113*tf.lnT9)

    rate_eval.be7__li7__weak__electron_capture = rate

@numba.njit()
def b12__c12__weak__wc17(rate_eval, tf):
    # b12 --> c12
    rate = 0.0

    # wc17w
    rate += np.exp(  3.53556)

    rate_eval.b12__c12__weak__wc17 = rate

@numba.njit()
def c11__b11__weak__wc12(rate_eval, tf):
    # c11 --> b11
    rate = 0.0

    # wc12w
    rate += np.exp(  -7.47312)

    rate_eval.c11__b11__weak__wc12 = rate

@numba.njit()
def c14__n14__weak__wc12(rate_eval, tf):
    # c14 --> n14
    rate = 0.0

    # wc12w
    rate += np.exp(  -26.2827)

    rate_eval.c14__n14__weak__wc12 = rate

@numba.njit()
def n12__c12__weak__wc12(rate_eval, tf):
    # n12 --> c12
    rate = 0.0

    # wc12w
    rate += np.exp(  4.14335)

    rate_eval.n12__c12__weak__wc12 = rate

@numba.njit()
def n13__c13__weak__wc12(rate_eval, tf):
    # n13 --> c13
    rate = 0.0

    # wc12w
    rate += np.exp(  -6.7601)

    rate_eval.n13__c13__weak__wc12 = rate

@numba.njit()
def o14__n14__weak__wc12(rate_eval, tf):
    # o14 --> n14
    rate = 0.0

    # wc12w
    rate += np.exp(  -4.62354)

    rate_eval.o14__n14__weak__wc12 = rate

@numba.njit()
def o15__n15__weak__wc12(rate_eval, tf):
    # o15 --> n15
    rate = 0.0

    # wc12w
    rate += np.exp(  -5.17053)

    rate_eval.o15__n15__weak__wc12 = rate

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
def li8__n_li7(rate_eval, tf):
    # li8 --> n + li7
    rate = 0.0

    # ks03 
    rate += np.exp(  25.783 + -23.559*tf.T9i + -3.24071*tf.T913i + 10.225*tf.T913
                  + -0.540218*tf.T9 + 0.0265361*tf.T953 + -2.11209*tf.lnT9)

    rate_eval.li8__n_li7 = rate

@numba.njit()
def li8__he4_he4__weak__wc12(rate_eval, tf):
    # li8 --> he4 + he4
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.19216)

    rate_eval.li8__he4_he4__weak__wc12 = rate

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
def b8__p_be7(rate_eval, tf):
    # b8 --> p + be7
    rate = 0.0

    # nacrn
    rate += np.exp(  35.8138 + -1.58982*tf.T9i + -10.264*tf.T913i + -0.203472*tf.T913
                  + 0.121083*tf.T9 + -0.00700063*tf.T953 + 0.833333*tf.lnT9)
    # nacrr
    rate += np.exp(  31.0163 + -8.93482*tf.T9i)

    rate_eval.b8__p_be7 = rate

@numba.njit()
def b8__he4_he4__weak__wc12(rate_eval, tf):
    # b8 --> he4 + he4
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.105148)

    rate_eval.b8__he4_he4__weak__wc12 = rate

@numba.njit()
def b10__p_be9(rate_eval, tf):
    # b10 --> p + be9
    rate = 0.0

    # nacrr
    rate += np.exp(  37.9538 + -87.9663*tf.T9i)
    # nacrr
    rate += np.exp(  30.6751 + -79.0223*tf.T9i)
    # nacrn
    rate += np.exp(  39.2789 + -76.4272*tf.T9i + -10.361*tf.T913i + 0.695179*tf.T913
                  + 0.342365*tf.T9 + -0.356569*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.b10__p_be9 = rate

@numba.njit()
def b10__he4_li6(rate_eval, tf):
    # b10 --> he4 + li6
    rate = 0.0

    # cf88r
    rate += np.exp(  24.5212 + -55.4692*tf.T9i + 3.33334*tf.T913i + 3.25335*tf.T913
                  + 0.374434*tf.T9 + -0.0706244*tf.T953)
    # cf88n
    rate += np.exp(  38.6952 + -51.7561*tf.T9i + -18.79*tf.T913i + 0.234225*tf.T913
                  + 3.23344*tf.T9 + -1.14529*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.b10__he4_li6 = rate

@numba.njit()
def b11__n_b10(rate_eval, tf):
    # b11 --> n + b10
    rate = 0.0

    #  wagn
    rate += np.exp(  35.2227 + -132.928*tf.T9i + 2.65756e-10*tf.T913i + -9.63588e-10*tf.T913
                  + 1.07466e-10*tf.T9 + -9.87569e-12*tf.T953 + 1.5*tf.lnT9)

    rate_eval.b11__n_b10 = rate

@numba.njit()
def b11__he4_li7(rate_eval, tf):
    # b11 --> he4 + li7
    rate = 0.0

    # nacrr
    rate += np.exp(  30.2249 + -103.501*tf.T9i)
    # nacrn
    rate += np.exp(  42.8425 + -100.541*tf.T9i + -19.163*tf.T913i + 0.0587651*tf.T913
                  + 0.773338*tf.T9 + -0.201519*tf.T953 + 0.833333*tf.lnT9)
    # nacrr
    rate += np.exp(  35.1078 + -106.983*tf.T9i
                  + 0.190698*tf.T9)

    rate_eval.b11__he4_li7 = rate

@numba.njit()
def b12__n_b11(rate_eval, tf):
    # b12 --> n + b11
    rate = 0.0

    # bb92n
    rate += np.exp(  30.4545 + -39.1015*tf.T9i + 2.18692e-11*tf.T913i + -4.50377e-11*tf.T913
                  + 2.23428e-12*tf.T9 + -5.80147e-14*tf.T953 + 1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  22.2692 + -39.2822*tf.T9i + -5.3754*tf.T913i + 15.15*tf.T913
                  + -0.151605*tf.T9 + -0.0619668*tf.T953 + -5.85999*tf.lnT9)

    rate_eval.b12__n_b11 = rate

@numba.njit()
def b12__he4_li8(rate_eval, tf):
    # b12 --> he4 + li8
    rate = 0.0

    # fkthr
    rate += np.exp(  62.5586 + -117.088*tf.T9i + 19.45*tf.T913i + -57.5572*tf.T913
                  + 3.09288*tf.T9 + -0.164411*tf.T953 + 28.5775*tf.lnT9)

    rate_eval.b12__he4_li8 = rate

@numba.njit()
def c11__p_b10(rate_eval, tf):
    # c11 --> p + b10
    rate = 0.0

    # nacrr
    rate += np.exp(  11.0036 + -105.289*tf.T9i + 105.797*tf.T913i + -86.6739*tf.T913
                  + 1.95564*tf.T9 + -0.0113213*tf.T953 + 62.4579*tf.lnT9)
    # nacrn
    rate += np.exp(  64.5157 + -100.831*tf.T9i + -13.4089*tf.T913i + -66.1173*tf.T913
                  + 163.12*tf.T9 + -250.002*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.c11__p_b10 = rate

@numba.njit()
def c11__he4_be7(rate_eval, tf):
    # c11 --> he4 + be7
    rate = 0.0

    # nacrr
    rate += np.exp(  36.2917 + -97.7214*tf.T9i + 0.685881*tf.T913
                  + -0.697071*tf.T9 + 0.13274*tf.T953)
    # nacrr
    rate += np.exp(  33.8476 + -94.0424*tf.T9i)
    # nacrn
    rate += np.exp(  48.6414 + -87.5443*tf.T9i + -23.214*tf.T913i + -3.74943*tf.T913
                  + 1.23242*tf.T9 + -0.195665*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.c11__he4_be7 = rate

@numba.njit()
def c12__n_c11(rate_eval, tf):
    # c12 --> n + c11
    rate = 0.0

    # bb92n
    rate += np.exp(  35.3287 + -217.262*tf.T9i + 1.18875e-10*tf.T913i + -3.10977e-10*tf.T913
                  + 2.0423e-11*tf.T9 + -6.59293e-13*tf.T953 + 1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  38.8258 + -222.832*tf.T9i + 1.38808e-09*tf.T913i + -3.66292e-09*tf.T913
                  + 2.41904e-10*tf.T9 + -7.82223e-12*tf.T953)
    # bb92r
    rate += np.exp(  33.0631 + -218.179*tf.T9i + -5.21741e-12*tf.T913i + 2.32794e-11*tf.T913
                  + -2.03119e-12*tf.T9 + 7.09452e-14*tf.T953)

    rate_eval.c12__n_c11 = rate

@numba.njit()
def c12__p_b11(rate_eval, tf):
    # c12 --> p + b11
    rate = 0.0

    # nw00r
    rate += np.exp(  33.6351 + -186.885*tf.T9i)
    # nw00n
    rate += np.exp(  50.5262 + -185.173*tf.T9i + -12.095*tf.T913i + -6.68421*tf.T913
                  + -0.0148736*tf.T9 + 0.0364288*tf.T953 + 2.83333*tf.lnT9)
    # nw00n
    rate += np.exp(  43.578 + -185.173*tf.T9i + -12.095*tf.T913i + -1.95046*tf.T913
                  + 9.56928*tf.T9 + -10.0637*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.c12__p_b11 = rate

@numba.njit()
def c13__n_c12(rate_eval, tf):
    # c13 --> n + c12
    rate = 0.0

    # ks03 
    rate += np.exp(  30.8808 + -57.4077*tf.T9i + 1.49573*tf.T913i + -0.841102*tf.T913
                  + 0.0340543*tf.T9 + -0.0026392*tf.T953 + 3.1662*tf.lnT9)

    rate_eval.c13__n_c12 = rate

@numba.njit()
def c14__n_c13(rate_eval, tf):
    # c14 --> n + c13
    rate = 0.0

    # ks03 
    rate += np.exp(  59.5926 + -95.0156*tf.T9i + 18.3578*tf.T913i + -46.5786*tf.T913
                  + 2.58472*tf.T9 + -0.118622*tf.T953 + 21.4142*tf.lnT9)

    rate_eval.c14__n_c13 = rate

@numba.njit()
def n12__p_c11(rate_eval, tf):
    # n12 --> p + c11
    rate = 0.0

    # gl07r
    rate += np.exp(  29.6539 + -11.1452*tf.T9i)
    # gl07n
    rate += np.exp(  37.9547 + -6.97916*tf.T9i + -13.6275*tf.T913i
                  + 0.722849*tf.T9 + 0.833333*tf.lnT9)

    rate_eval.n12__p_c11 = rate

@numba.njit()
def n13__p_c12(rate_eval, tf):
    # n13 --> p + c12
    rate = 0.0

    # ls09r
    rate += np.exp(  40.4354 + -26.326*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9)
    # ls09n
    rate += np.exp(  40.0408 + -22.5475*tf.T9i + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.n13__p_c12 = rate

@numba.njit()
def n14__n_n13(rate_eval, tf):
    # n14 --> n + n13
    rate = 0.0

    # wiesr
    rate += np.exp(  19.5584 + -125.474*tf.T9i + 9.44873e-10*tf.T913i + -2.33713e-09*tf.T913
                  + 1.97507e-10*tf.T9 + -1.49747e-11*tf.T953)
    # wiesn
    rate += np.exp(  37.1268 + -122.484*tf.T9i + 1.72241e-10*tf.T913i + -5.62522e-10*tf.T913
                  + 5.59212e-11*tf.T9 + -4.6549e-12*tf.T953 + 1.5*tf.lnT9)

    rate_eval.n14__n_n13 = rate

@numba.njit()
def n14__p_c13(rate_eval, tf):
    # n14 --> p + c13
    rate = 0.0

    # nacrr
    rate += np.exp(  37.1528 + -93.4071*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953)
    # nacrr
    rate += np.exp(  38.3716 + -101.18*tf.T9i)
    # nacrn
    rate += np.exp(  41.7046 + -87.6256*tf.T9i + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.n14__p_c13 = rate

@numba.njit()
def n15__n_n14(rate_eval, tf):
    # n15 --> n + n14
    rate = 0.0

    # ks03 
    rate += np.exp(  34.1728 + -125.726*tf.T9i + 1.396*tf.T913i + -3.47552*tf.T913
                  + 0.351773*tf.T9 + -0.0229344*tf.T953 + 2.52161*tf.lnT9)

    rate_eval.n15__n_n14 = rate

@numba.njit()
def n15__p_c14(rate_eval, tf):
    # n15 --> p + c14
    rate = 0.0

    # il10r
    rate += np.exp(  40.0115 + -119.975*tf.T9i + -10.658*tf.T913i + 1.73644*tf.T913
                  + -0.350498*tf.T9 + 0.0279902*tf.T953)
    # il10n
    rate += np.exp(  43.0281 + -118.452*tf.T9i + -13.9619*tf.T913i + -4.34315*tf.T913
                  + 6.64922*tf.T9 + -3.22592*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.n15__p_c14 = rate

@numba.njit()
def o14__p_n13(rate_eval, tf):
    # o14 --> p + n13
    rate = 0.0

    # lg06r
    rate += np.exp(  35.2849 + -59.8313*tf.T9i + 1.57122*tf.T913i)
    # lg06n
    rate += np.exp(  42.4234 + -53.7053*tf.T9i + -15.1676*tf.T913i + 0.0955166*tf.T913
                  + 3.0659*tf.T9 + -0.507339*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.o14__p_n13 = rate

@numba.njit()
def o15__n_o14(rate_eval, tf):
    # o15 --> n + o14
    rate = 0.0

    # rpsmr
    rate += np.exp(  32.7811 + -153.419*tf.T9i + -1.38986*tf.T913i + 1.74662*tf.T913
                  + -0.0276897*tf.T9 + 0.00321014*tf.T953 + 0.438778*tf.lnT9)

    rate_eval.o15__n_o14 = rate

@numba.njit()
def o15__p_n14(rate_eval, tf):
    # o15 --> p + n14
    rate = 0.0

    # im05r
    rate += np.exp(  30.7435 + -89.5667*tf.T9i
                  + 1.5682*tf.lnT9)
    # im05r
    rate += np.exp(  31.6622 + -87.6737*tf.T9i)
    # im05n
    rate += np.exp(  44.1246 + -84.6757*tf.T9i + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 1.83333*tf.lnT9)
    # im05n
    rate += np.exp(  41.0177 + -84.6757*tf.T9i + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.o15__p_n14 = rate

@numba.njit()
def o16__n_o15(rate_eval, tf):
    # o16 --> n + o15
    rate = 0.0

    # rpsmr
    rate += np.exp(  32.3869 + -181.759*tf.T9i + -1.11761*tf.T913i + 1.0167*tf.T913
                  + 0.0449976*tf.T9 + -0.00204682*tf.T953 + 0.710783*tf.lnT9)

    rate_eval.o16__n_o15 = rate

@numba.njit()
def o16__p_n15(rate_eval, tf):
    # o16 --> p + n15
    rate = 0.0

    # li10r
    rate += np.exp(  38.8465 + -150.962*tf.T9i
                  + 0.0459037*tf.T9)
    # li10r
    rate += np.exp(  30.8927 + -143.656*tf.T9i)
    # li10n
    rate += np.exp(  44.3197 + -140.732*tf.T9i + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.o16__p_n15 = rate

@numba.njit()
def o16__he4_c12(rate_eval, tf):
    # o16 --> he4 + c12
    rate = 0.0

    # nac2 
    rate += np.exp(  279.295 + -84.9515*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 138.803*tf.lnT9)
    # nac2 
    rate += np.exp(  94.3131 + -84.503*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 71.8554*tf.lnT9)

    rate_eval.o16__he4_c12 = rate

@numba.njit()
def li6__n_p_he4(rate_eval, tf):
    # li6 --> n + p + he4
    rate = 0.0

    # cf88r
    rate += np.exp(  33.4196 + -62.2896*tf.T9i + 1.44987*tf.T913i + -1.42759*tf.T913
                  + 0.0454035*tf.T9 + 0.00471161*tf.T953 + 2.0*tf.lnT9)

    rate_eval.li6__n_p_he4 = rate

@numba.njit()
def be9__n_he4_he4(rate_eval, tf):
    # be9 --> n + he4 + he4
    rate = 0.0

    # ac12r
    rate += np.exp(  38.6902 + -19.2792*tf.T9i + -1.56673*tf.T913i + -5.43497*tf.T913
                  + 0.673807*tf.T9 + -0.041014*tf.T953 + 1.5*tf.lnT9)
    # ac12n
    rate += np.exp(  37.273 + -18.2597*tf.T9i + -13.3317*tf.T913i + 13.2237*tf.T913
                  + -9.06339*tf.T9 + 2.33333*tf.lnT9)

    rate_eval.be9__n_he4_he4 = rate

@numba.njit()
def c12__he4_he4_he4(rate_eval, tf):
    # c12 --> he4 + he4 + he4
    rate = 0.0

    # fy05n
    rate += np.exp(  45.7734 + -84.4227*tf.T9i + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + 1.66667*tf.lnT9)
    # fy05r
    rate += np.exp(  22.394 + -88.5493*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -10.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  34.9561 + -85.4472*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + 0.83333*tf.lnT9)

    rate_eval.c12__he4_he4_he4 = rate

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

    #Alterrate
    '''
    if (tf.T9<2.5):
        rate=((0.094614248-4.9273133*tf.T9+99.358965*tf.T9*tf.T9-989.81236*tf.T9*tf.T9*tf.T9+4368.45*pow(tf.T9,4.)+931.93597*pow(tf.T9,5.)
                -391.07855*pow(tf.T9,6.)+159.23101*pow(tf.T9,7.)-34.407594*pow(tf.T9,8.)+3.3919004*pow(tf.T9,9.)
                +0.017556217*pow(tf.T9,10.)-0.036253427*pow(tf.T9,11.)+0.0031118827*pow(tf.T9,12.)
                -0.00008714468*pow(tf.T9,13.))*pow(tf.T9,-1./2.))/(np.exp(8.4e-7*tf.T9)*pow(1.+1.78616593*tf.T9,3.))
    
    else: rate=807.406
    '''

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
def he4_li6__b10(rate_eval, tf):
    # li6 + he4 --> b10
    rate = 0.0

    # cf88r
    rate += np.exp(  1.04267 + -3.71313*tf.T9i + 3.33334*tf.T913i + 3.25335*tf.T913
                  + 0.374434*tf.T9 + -0.0706244*tf.T953 + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  15.2167 + -18.79*tf.T913i + 0.234225*tf.T913
                  + 3.23344*tf.T9 + -1.14529*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_li6__b10 = rate

@numba.njit()
def n_li7__li8(rate_eval, tf):
    # li7 + n --> li8
    rate = 0.0

    # ks03 
    rate += np.exp(  2.50071 + 0.0248641*tf.T9i + -3.24071*tf.T913i + 10.225*tf.T913
                  + -0.540218*tf.T9 + 0.0265361*tf.T953 + -3.61209*tf.lnT9)

    rate_eval.n_li7__li8 = rate

@numba.njit()
def he4_li7__b11(rate_eval, tf):
    # li7 + he4 --> b11
    rate = 0.0

    # nacrr
    rate += np.exp(  10.6937 + -6.44203*tf.T9i
                  + 0.190698*tf.T9 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  5.81084 + -2.95915*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  18.4284 + -19.163*tf.T913i + 0.0587651*tf.T913
                  + 0.773338*tf.T9 + -0.201519*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_li7__b11 = rate

@numba.njit()
def he4_li8__b12(rate_eval, tf):
    # li8 + he4 --> b12
    rate = 0.0

    # fkthr
    rate += np.exp(  37.5639 + -1.04393*tf.T9i + 19.45*tf.T913i + -57.5572*tf.T913
                  + 3.09288*tf.T9 + -0.164411*tf.T953 + 27.0775*tf.lnT9)

    rate_eval.he4_li8__b12 = rate

@numba.njit()
def p_be7__b8(rate_eval, tf):
    # be7 + p --> b8
    rate = 0.0

    # nacrn
    rate += np.exp(  12.5315 + -10.264*tf.T913i + -0.203472*tf.T913
                  + 0.121083*tf.T9 + -0.00700063*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  7.73399 + -7.345*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_be7__b8 = rate

@numba.njit()
def he4_be7__c11(rate_eval, tf):
    # be7 + he4 --> c11
    rate = 0.0

    # nacrr
    rate += np.exp(  11.8776 + -10.177*tf.T9i + 0.685881*tf.T913
                  + -0.697071*tf.T9 + 0.13274*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  9.43348 + -6.498*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  24.2273 + -23.214*tf.T913i + -3.74943*tf.T913
                  + 1.23242*tf.T9 + -0.195665*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_be7__c11 = rate

@numba.njit()
def p_be9__b10(rate_eval, tf):
    # be9 + p --> b10
    rate = 0.0

    # nacrr
    rate += np.exp(  14.9657 + -11.5391*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  7.68698 + -2.59506*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  16.2908 + -10.361*tf.T913i + 0.695179*tf.T913
                  + 0.342365*tf.T9 + -0.356569*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_be9__b10 = rate

@numba.njit()
def n_b10__b11(rate_eval, tf):
    # b10 + n --> b11
    rate = 0.0

    #  wagn
    rate += np.exp(  11.1004 + -1.9027e-12*tf.T9i + 2.65756e-10*tf.T913i + -9.63588e-10*tf.T913
                  + 1.07466e-10*tf.T9 + -9.87569e-12*tf.T953 + 3.12603e-10*tf.lnT9)

    rate_eval.n_b10__b11 = rate

@numba.njit()
def p_b10__c11(rate_eval, tf):
    # b10 + p --> c11
    rate = 0.0

    # nacrr
    rate += np.exp(  -13.1188 + -4.45749*tf.T9i + 105.797*tf.T913i + -86.6739*tf.T913
                  + 1.95564*tf.T9 + -0.0113213*tf.T953 + 60.9579*tf.lnT9)
    # nacrn
    rate += np.exp(  40.3933 + -13.4089*tf.T913i + -66.1173*tf.T913
                  + 163.12*tf.T9 + -250.002*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b10__c11 = rate

@numba.njit()
def n_b11__b12(rate_eval, tf):
    # b11 + n --> b12
    rate = 0.0

    # bb92r
    rate += np.exp(  -1.59362 + -0.180671*tf.T9i + -5.3754*tf.T913i + 15.15*tf.T913
                  + -0.151605*tf.T9 + -0.0619668*tf.T953 + -7.35999*tf.lnT9)
    # bb92n
    rate += np.exp(  6.59167 + -2.1598e-13*tf.T9i + 2.18692e-11*tf.T913i + -4.50377e-11*tf.T913
                  + 2.23428e-12*tf.T9 + -5.80147e-14*tf.T953 + 2.002e-11*tf.lnT9)

    rate_eval.n_b11__b12 = rate

@numba.njit()
def p_b11__c12(rate_eval, tf):
    # b11 + p --> c12
    rate = 0.0

    # nw00r
    rate += np.exp(  8.67352 + -1.71197*tf.T9i
                  + -1.5*tf.lnT9)
    # nw00n
    rate += np.exp(  25.5647 + -12.095*tf.T913i + -6.68421*tf.T913
                  + -0.0148736*tf.T9 + 0.0364288*tf.T953 + 1.33333*tf.lnT9)
    # nw00n
    rate += np.exp(  18.6165 + -12.095*tf.T913i + -1.95046*tf.T913
                  + 9.56928*tf.T9 + -10.0637*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b11__c12 = rate

@numba.njit()
def n_c11__c12(rate_eval, tf):
    # c11 + n --> c12
    rate = 0.0

    # bb92r
    rate += np.exp(  13.8643 + -5.57951*tf.T9i + 1.38808e-09*tf.T913i + -3.66292e-09*tf.T913
                  + 2.41904e-10*tf.T9 + -7.82223e-12*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  8.10155 + -0.926514*tf.T9i + -5.21741e-12*tf.T913i + 2.32794e-11*tf.T913
                  + -2.03119e-12*tf.T9 + 7.09452e-14*tf.T953 + -1.5*tf.lnT9)
    # bb92n
    rate += np.exp(  10.3672 + -0.0095137*tf.T9i + 1.18875e-10*tf.T913i + -3.10977e-10*tf.T913
                  + 2.0423e-11*tf.T9 + -6.59293e-13*tf.T953)

    rate_eval.n_c11__c12 = rate

@numba.njit()
def p_c11__n12(rate_eval, tf):
    # c11 + p --> n12
    rate = 0.0

    # gl07r
    rate += np.exp(  5.791 + -4.16602*tf.T9i
                  + -1.5*tf.lnT9)
    # gl07n
    rate += np.exp(  14.0918 + -13.6275*tf.T913i
                  + 0.722849*tf.T9 + -0.666667*tf.lnT9)

    rate_eval.p_c11__n12 = rate

@numba.njit()
def n_c12__c13(rate_eval, tf):
    # c12 + n --> c13
    rate = 0.0

    # ks03 
    rate += np.exp(  7.98821 + -0.00836732*tf.T9i + 1.49573*tf.T913i + -0.841102*tf.T913
                  + 0.0340543*tf.T9 + -0.0026392*tf.T953 + 1.6662*tf.lnT9)

    rate_eval.n_c12__c13 = rate

@numba.njit()
def p_c12__n13(rate_eval, tf):
    # c12 + p --> n13
    rate = 0.0

    # ls09n
    rate += np.exp(  17.1482 + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + -0.666667*tf.lnT9)
    # ls09r
    rate += np.exp(  17.5428 + -3.77849*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9 + -1.5*tf.lnT9)

    rate_eval.p_c12__n13 = rate

@numba.njit()
def he4_c12__o16(rate_eval, tf):
    # c12 + he4 --> o16
    rate = 0.0

    # nac2 
    rate += np.exp(  254.634 + -1.84097*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 137.303*tf.lnT9)
    # nac2 
    rate += np.exp(  69.6526 + -1.39254*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 70.3554*tf.lnT9)

    rate_eval.he4_c12__o16 = rate

@numba.njit()
def n_c13__c14(rate_eval, tf):
    # c13 + n --> c14
    rate = 0.0

    # ks03 
    rate += np.exp(  35.3048 + -0.133687*tf.T9i + 18.3578*tf.T913i + -46.5786*tf.T913
                  + 2.58472*tf.T9 + -0.118622*tf.T953 + 19.9142*tf.lnT9)

    rate_eval.n_c13__c14 = rate

@numba.njit()
def p_c13__n14(rate_eval, tf):
    # c13 + p --> n14
    rate = 0.0

    # nacrr
    rate += np.exp(  15.1825 + -13.5543*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  18.5155 + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  13.9637 + -5.78147*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_c13__n14 = rate

@numba.njit()
def p_c14__n15(rate_eval, tf):
    # c14 + p --> n15
    rate = 0.0

    # il10n
    rate += np.exp(  20.119 + -13.9619*tf.T913i + -4.34315*tf.T913
                  + 6.64922*tf.T9 + -3.22592*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  17.1024 + -1.52341*tf.T9i + -10.658*tf.T913i + 1.73644*tf.T913
                  + -0.350498*tf.T9 + 0.0279902*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_c14__n15 = rate

@numba.njit()
def n_n13__n14(rate_eval, tf):
    # n13 + n --> n14
    rate = 0.0

    # wiesr
    rate += np.exp(  -3.63074 + -2.99547*tf.T9i + 9.44873e-10*tf.T913i + -2.33713e-09*tf.T913
                  + 1.97507e-10*tf.T9 + -1.49747e-11*tf.T953 + -1.5*tf.lnT9)
    # wiesn
    rate += np.exp(  13.9377 + -0.0054652*tf.T9i + 1.72241e-10*tf.T913i + -5.62522e-10*tf.T913
                  + 5.59212e-11*tf.T9 + -4.6549e-12*tf.T953)

    rate_eval.n_n13__n14 = rate

@numba.njit()
def p_n13__o14(rate_eval, tf):
    # n13 + p --> o14
    rate = 0.0

    # lg06r
    rate += np.exp(  10.9971 + -6.12602*tf.T9i + 1.57122*tf.T913i
                  + -1.5*tf.lnT9)
    # lg06n
    rate += np.exp(  18.1356 + -15.1676*tf.T913i + 0.0955166*tf.T913
                  + 3.0659*tf.T9 + -0.507339*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_n13__o14 = rate

@numba.njit()
def n_n14__n15(rate_eval, tf):
    # n14 + n --> n15
    rate = 0.0

    # ks03 
    rate += np.exp(  10.1651 + -0.0114078*tf.T9i + 1.396*tf.T913i + -3.47552*tf.T913
                  + 0.351773*tf.T9 + -0.0229344*tf.T953 + 1.02161*tf.lnT9)

    rate_eval.n_n14__n15 = rate

@numba.njit()
def p_n14__o15(rate_eval, tf):
    # n14 + p --> o15
    rate = 0.0

    # im05n
    rate += np.exp(  17.01 + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + -0.666667*tf.lnT9)
    # im05r
    rate += np.exp(  6.73578 + -4.891*tf.T9i
                  + 0.0682*tf.lnT9)
    # im05r
    rate += np.exp(  7.65444 + -2.998*tf.T9i
                  + -1.5*tf.lnT9)
    # im05n
    rate += np.exp(  20.1169 + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 0.333333*tf.lnT9)

    rate_eval.p_n14__o15 = rate

@numba.njit()
def p_n15__o16(rate_eval, tf):
    # n15 + p --> o16
    rate = 0.0

    # li10n
    rate += np.exp(  20.0176 + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + -0.666667*tf.lnT9)
    # li10r
    rate += np.exp(  14.5444 + -10.2295*tf.T9i
                  + 0.0459037*tf.T9 + -1.5*tf.lnT9)
    # li10r
    rate += np.exp(  6.59056 + -2.92315*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_n15__o16 = rate

@numba.njit()
def n_o14__o15(rate_eval, tf):
    # o14 + n --> o15
    rate = 0.0

    # rpsmr
    rate += np.exp(  9.87196 + 0.0160481*tf.T9i + -1.38986*tf.T913i + 1.74662*tf.T913
                  + -0.0276897*tf.T9 + 0.00321014*tf.T953 + -1.06122*tf.lnT9)

    rate_eval.n_o14__o15 = rate

@numba.njit()
def n_o15__o16(rate_eval, tf):
    # o15 + n --> o16
    rate = 0.0

    # rpsmr
    rate += np.exp(  8.08476 + 0.0135346*tf.T9i + -1.11761*tf.T913i + 1.0167*tf.T913
                  + 0.0449976*tf.T9 + -0.00204682*tf.T953 + -0.789217*tf.lnT9)

    rate_eval.n_o15__o16 = rate

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
def he4_li6__p_be9(rate_eval, tf):
    # li6 + he4 --> p + be9
    rate = 0.0

    # cf88r
    rate += np.exp(  19.8324 + -29.8312*tf.T9i
                  + -0.75*tf.lnT9)
    # cf88r
    rate += np.exp(  19.4366 + -27.7172*tf.T9i
                  + -1.0*tf.lnT9)
    # cf88n
    rate += np.exp(  25.5847 + -24.6712*tf.T9i + -10.359*tf.T913i + 0.102577*tf.T913
                  + 4.43544*tf.T9 + -5.97105*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_li6__p_be9 = rate

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
    '''
    #Alterrate
    if (tf.T9<2.5):
        rate=((-8.9654123e7-2.5851582e8*tf.T9-2.6831252e7*tf.T9*tf.T9+3.8691673e8*pow(tf.T9,1./3.)+4.9721269e8*pow(tf.T9,2./3.)
                +2.6444808e7*pow(tf.T9,4./3.)-1.2946419e6*pow(tf.T9,5./3.)-1.0941088e8*pow(tf.T9,7./3.)
                +9.9899564e7*pow(tf.T9,8./3.))*pow(tf.T9,-2./3.))/np.exp(7.73389632*pow(tf.T9,-1./3.))
        rate+=np.exp(-1.137519e0*tf.T9*tf.T9-8.6256687*pow(tf.T9,-1./3.))*(3.0014189e7-1.8366119e8*tf.T9+1.7688138e9*tf.T9*tf.T9
                                                                -8.4772261e9*tf.T9*tf.T9*tf.T9+2.0237351e10*pow(tf.T9,4.)
                                                                -1.9650068e10*pow(tf.T9,5.)+7.9452762e8*pow(tf.T9,6.)
                                                                +1.3132468e10*pow(tf.T9,7.)-8.209351e9*pow(tf.T9,8.)
                                                                -9.1099236e8*pow(tf.T9,9.)+2.7814079e9*pow(tf.T9,10.)
                                                                -1.0785293e9*pow(tf.T9,11.)
                                                                +1.3993392e8*pow(tf.T9,12.))*pow(tf.T9,-2./3.)
    else:
        rate=1.53403e6
        rate+=84516.7

    '''


    rate_eval.p_li7__he4_he4 = rate

@numba.njit()
def d_li7__p_li8(rate_eval, tf):
    # li7 + d --> p + li8
    rate = 0.0

    # mafor
    rate += np.exp(  20.5381 + -6.998*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.d_li7__p_li8 = rate

@numba.njit()
def t_li7__n_be9(rate_eval, tf):
    # li7 + t --> n + be9
    rate = 0.0

    # bk91 
    rate += np.exp(  34.634 + -11.333*tf.T913i + -7.3964*tf.T913
                  + 0.947759*tf.T9 + -0.0839008*tf.T953 + 0.333333*tf.lnT9)
    # bk91 
    rate += np.exp(  30.2619 + -11.333*tf.T913i + -0.170964*tf.T913
                  + -6.30572*tf.T9 + 1.2248*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_li7__n_be9 = rate

@numba.njit()
def t_li7__d_li8(rate_eval, tf):
    # li7 + t --> d + li8
    rate = 0.0

    # hi09r
    rate += np.exp(  19.7466 + -52.5346*tf.T9i
                  + -0.624*tf.lnT9)
    # hi09n
    rate += np.exp(  27.3104 + -49.0246*tf.T9i + -19.72*tf.T913i + 0.264846*tf.T913
                  + -0.0181997*tf.T9 + 0.00188655*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.t_li7__d_li8 = rate

@numba.njit()
def he4_li7__n_b10(rate_eval, tf):
    # li7 + he4 --> n + b10
    rate = 0.0

    # cf88n
    rate += np.exp(  19.7521 + -32.3766*tf.T9i)

    rate_eval.he4_li7__n_b10 = rate

@numba.njit()
def p_li8__d_li7(rate_eval, tf):
    # li8 + p --> d + li7
    rate = 0.0

    # mafor
    rate += np.exp(  21.5598 + -4.78119*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_li8__d_li7 = rate

@numba.njit()
def d_li8__n_be9(rate_eval, tf):
    # li8 + d --> n + be9
    rate = 0.0

    # mafon
    rate += np.exp(  26.4 + -5.94754e-12*tf.T9i + -10.259*tf.T913i + -2.53711e-09*tf.T913
                  + 2.5234e-10*tf.T9 + -2.10055e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_li8__n_be9 = rate

@numba.njit()
def d_li8__t_li7(rate_eval, tf):
    # li8 + d --> t + li7
    rate = 0.0

    # hi09r
    rate += np.exp(  19.5259 + -3.51*tf.T9i
                  + -0.624*tf.lnT9)
    # hi09n
    rate += np.exp(  27.0897 + -19.72*tf.T913i + 0.264846*tf.T913
                  + -0.0181997*tf.T9 + 0.00188655*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_li8__t_li7 = rate

@numba.njit()
def he4_li8__n_b11(rate_eval, tf):
    # li8 + he4 --> n + b11
    rate = 0.0

    # ld11 
    rate += np.exp(  35.63 + -2.51*tf.T9i + -32.33*tf.T913i + 13.17*tf.T913
                  + -0.01816*tf.T9 + -0.02032*tf.T953 + -12.38*tf.lnT9)

    rate_eval.he4_li8__n_b11 = rate

@numba.njit()
def n_be7__p_li7(rate_eval, tf):
    # be7 + n --> p + li7
    rate = 0.0

    # db18 
    rate += np.exp(  21.7899 + 0.000728098*tf.T9i + -0.30254*tf.T913i + -0.3602*tf.T913
                  + 0.17472*tf.T9 + -0.0223*tf.T953 + -0.4581*tf.lnT9)
    #Alterrate
    '''
    if(tf.T9<2.5):
        rate=(6.8423032e9+1.7674863e10*tf.T9+2.6622006e9*tf.T9*tf.T9-3.3561608e8*tf.T9*tf.T9*tf.T9-5.9309139e6*pow(tf.T9,4.)
        -1.4987996e10*np.sqrt(tf.T9)-1.0576906e10*pow(tf.T9,3./2.)+2.7447598e8*pow(tf.T9,5./2.)
        +7.6425157e7*pow(tf.T9,7./2.)-2.282944e7*pow(tf.T9,-3./2.) / np.exp(0.050351813/tf.T9))

    else: rate=1.28039e9
    '''
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
def he4_be7__p_b10(rate_eval, tf):
    # be7 + he4 --> p + b10
    rate = 0.0

    # nacrr
    rate += np.exp(  -6.7467 + -13.8479*tf.T9i + 0.532995*tf.T913i + 22.8893*tf.T913
                  + -3.08149*tf.T9 + 0.218269*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  43.9213 + -13.2872*tf.T9i + -12.9754*tf.T913i + -44.3224*tf.T913
                  + 62.9626*tf.T9 + -49.5228*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_be7__p_b10 = rate

@numba.njit()
def n_be9__d_li8(rate_eval, tf):
    # be9 + n --> d + li8
    rate = 0.0

    # mafon
    rate += np.exp(  27.8917 + -170.148*tf.T9i + -10.259*tf.T913i + -2.53711e-09*tf.T913
                  + 2.5234e-10*tf.T9 + -2.10055e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_be9__d_li8 = rate

@numba.njit()
def n_be9__t_li7(rate_eval, tf):
    # be9 + n --> t + li7
    rate = 0.0

    # bk91 
    rate += np.exp(  35.9049 + -121.123*tf.T9i + -11.333*tf.T913i + -7.3964*tf.T913
                  + 0.947759*tf.T9 + -0.0839008*tf.T953 + 0.333333*tf.lnT9)
    # bk91 
    rate += np.exp(  31.5328 + -121.123*tf.T9i + -11.333*tf.T913i + -0.170964*tf.T913
                  + -6.30572*tf.T9 + 1.2248*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_be9__t_li7 = rate

@numba.njit()
def p_be9__he4_li6(rate_eval, tf):
    # be9 + p --> he4 + li6
    rate = 0.0

    # cf88r
    rate += np.exp(  19.927 + -3.046*tf.T9i
                  + -1.0*tf.lnT9)
    # cf88n
    rate += np.exp(  26.0751 + -10.359*tf.T913i + 0.102577*tf.T913
                  + 4.43544*tf.T9 + -5.97105*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  20.3228 + -5.16*tf.T9i
                  + -0.75*tf.lnT9)

    rate_eval.p_be9__he4_li6 = rate

@numba.njit()
def t_be9__n_b11(rate_eval, tf):
    # be9 + t --> n + b11
    rate = 0.0

    # bb92n
    rate += np.exp(  28.966 + -5.61387e-12*tf.T9i + -14.02*tf.T913i + -2.37143e-09*tf.T913
                  + 2.34748e-10*tf.T9 + -1.94782e-11*tf.T953 + -0.666667*tf.lnT9)
    # bb92r
    rate += np.exp(  18.6438 + -4.43*tf.T9i + -4.68045e-09*tf.T913i + 1.05992e-08*tf.T913
                  + -8.43414e-10*tf.T9 + 6.15011e-11*tf.T953 + -1.5*tf.lnT9)

    rate_eval.t_be9__n_b11 = rate

@numba.njit()
def he4_be9__n_c12(rate_eval, tf):
    # be9 + he4 --> n + c12
    rate = 0.0

    # cf88r
    rate += np.exp(  11.744 + -4.179*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88r
    rate += np.exp(  -1.48281 + -1.834*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88r
    rate += np.exp(  -9.51959 + -1.184*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  31.464 + -23.87*tf.T913i + 0.566698*tf.T913
                  + 44.0957*tf.T9 + -314.232*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  19.2962 + -12.732*tf.T9i)

    rate_eval.he4_be9__n_c12 = rate

@numba.njit()
def he4_be9__p_b12(rate_eval, tf):
    # be9 + he4 --> p + b12
    rate = 0.0

    #  wagn
    rate += np.exp(  24.7841 + -79.9133*tf.T9i + -12.12*tf.T913i + -1.22842e-09*tf.T913
                  + 1.34011e-10*tf.T9 + -1.21412e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_be9__p_b12 = rate

@numba.njit()
def he4_b8__p_c11(rate_eval, tf):
    # b8 + he4 --> p + c11
    rate = 0.0

    # wiesr
    rate += np.exp(  207.835 + -9.09757*tf.T9i + 381.949*tf.T913i + -608.493*tf.T913
                  + 34.7229*tf.T9 + -2.02465*tf.T953 + 297.922*tf.lnT9)
    # wiesr
    rate += np.exp(  13.8524 + -1.0296*tf.T9i + 3.19293*tf.T913i + -37.5025*tf.T913
                  + 4.4781*tf.T9 + -0.401587*tf.T953 + 8.35133*tf.lnT9)

    rate_eval.he4_b8__p_c11 = rate

@numba.njit()
def n_b10__he4_li7(rate_eval, tf):
    # b10 + n --> he4 + li7
    rate = 0.0

    # cf88n
    rate += np.exp(  20.0438)

    rate_eval.n_b10__he4_li7 = rate

@numba.njit()
def p_b10__he4_be7(rate_eval, tf):
    # b10 + p --> he4 + be7
    rate = 0.0

    # nacrr
    rate += np.exp(  -6.45503 + -0.560753*tf.T9i + 0.532995*tf.T913i + 22.8893*tf.T913
                  + -3.08149*tf.T9 + 0.218269*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  44.213 + -12.9754*tf.T913i + -44.3224*tf.T913
                  + 62.9626*tf.T9 + -49.5228*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b10__he4_be7 = rate

@numba.njit()
def he4_b10__n_n13(rate_eval, tf):
    # b10 + he4 --> n + n13
    rate = 0.0

    # cf88n
    rate += np.exp(  30.5042 + -27.8719*tf.T913i + -0.599503*tf.T913
                  + 0.122849*tf.T9 + -0.0393717*tf.T953 + -0.507333*tf.lnT9)

    rate_eval.he4_b10__n_n13 = rate

@numba.njit()
def he4_b10__p_c13(rate_eval, tf):
    # b10 + he4 --> p + c13
    rate = 0.0

    #  wagn
    rate += np.exp(  34.498 + -27.99*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.he4_b10__p_c13 = rate

@numba.njit()
def n_b11__t_be9(rate_eval, tf):
    # b11 + n --> t + be9
    rate = 0.0

    # bb92n
    rate += np.exp(  30.3129 + -110.928*tf.T9i + -14.02*tf.T913i + -2.37143e-09*tf.T913
                  + 2.34748e-10*tf.T9 + -1.94782e-11*tf.T953 + -0.666667*tf.lnT9)
    # bb92r
    rate += np.exp(  19.9907 + -115.358*tf.T9i + -4.68045e-09*tf.T913i + 1.05992e-08*tf.T913
                  + -8.43414e-10*tf.T9 + 6.15011e-11*tf.T953 + -1.5*tf.lnT9)

    rate_eval.n_b11__t_be9 = rate

@numba.njit()
def n_b11__he4_li8(rate_eval, tf):
    # b11 + n --> he4 + li8
    rate = 0.0

    # ld11 
    rate += np.exp(  36.7618 + -79.4768*tf.T9i + -32.33*tf.T913i + 13.17*tf.T913
                  + -0.01816*tf.T9 + -0.02032*tf.T953 + -12.38*tf.lnT9)

    rate_eval.n_b11__he4_li8 = rate

@numba.njit()
def p_b11__n_c11(rate_eval, tf):
    # b11 + p --> n + c11
    rate = 0.0

    # cf88n
    rate += np.exp(  18.8963 + -32.0748*tf.T9i)

    rate_eval.p_b11__n_c11 = rate

@numba.njit()
def he4_b11__n_n14(rate_eval, tf):
    # b11 + he4 --> n + n14
    rate = 0.0

    # cf88r
    rate += np.exp(  0.582216 + -2.827*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  29.5726 + -28.234*tf.T913i + -0.325987*tf.T913
                  + 30.135*tf.T9 + -78.4165*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  15.3084 + -8.596*tf.T9i
                  + 0.6*tf.lnT9)
    # cf88r
    rate += np.exp(  7.44425 + -5.178*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_b11__n_n14 = rate

@numba.njit()
def he4_b11__p_c14(rate_eval, tf):
    # b11 + he4 --> p + c14
    rate = 0.0

    # bb92r
    rate += np.exp(  -5.21398 + -2.868*tf.T9i + 2.62625e-09*tf.T913i + -6.58921e-09*tf.T913
                  + 5.62244e-10*tf.T9 + -4.28925e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  5.48852 + -5.147*tf.T9i + -5.81643e-09*tf.T913i + 1.24374e-08*tf.T913
                  + -9.55069e-10*tf.T9 + 6.81706e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  6.1942 + -5.157*tf.T9i + -2.8504e-09*tf.T913i + 5.85721e-09*tf.T913
                  + -4.34052e-10*tf.T9 + 3.01373e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92n
    rate += np.exp(  178.316 + -0.19519*tf.T9i + 4.28912*tf.T913i + -214.72*tf.T913
                  + 57.4073*tf.T9 + -25.5329*tf.T953 + 53.0473*tf.lnT9)
    # bb92r
    rate += np.exp(  15.4137 + -11.26*tf.T9i + -1.87598e-08*tf.T913i + 3.26423e-08*tf.T913
                  + -2.18782e-09*tf.T9 + 1.43323e-10*tf.T953 + 0.6*tf.lnT9)

    rate_eval.he4_b11__p_c14 = rate

@numba.njit()
def p_b12__n_c12(rate_eval, tf):
    # b12 + p --> n + c12
    rate = 0.0

    #  wagn
    rate += np.exp(  26.7197 + -3.77039e-12*tf.T9i + -12.12*tf.T913i + -1.83547e-09*tf.T913
                  + 2.0104e-10*tf.T9 + -1.82499e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b12__n_c12 = rate

@numba.njit()
def p_b12__he4_be9(rate_eval, tf):
    # b12 + p --> he4 + be9
    rate = 0.0

    #  wagn
    rate += np.exp(  26.0266 + -2.55368e-12*tf.T9i + -12.12*tf.T913i + -1.22842e-09*tf.T913
                  + 1.34011e-10*tf.T9 + -1.21412e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b12__he4_be9 = rate

@numba.njit()
def he4_b12__n_n15(rate_eval, tf):
    # b12 + he4 --> n + n15
    rate = 0.0

    #  wagn
    rate += np.exp(  35.6506 + -28.45*tf.T913i + 2.82044e-10*tf.T913
                  + -2.59013e-11*tf.T9 + 2.07893e-12*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_b12__n_n15 = rate

@numba.njit()
def n_c11__p_b11(rate_eval, tf):
    # c11 + n --> p + b11
    rate = 0.0

    # cf88n
    rate += np.exp(  18.8963)

    rate_eval.n_c11__p_b11 = rate

@numba.njit()
def p_c11__he4_b8(rate_eval, tf):
    # c11 + p --> he4 + b8
    rate = 0.0

    # wiesr
    rate += np.exp(  14.9842 + -86.975*tf.T9i + 3.19293*tf.T913i + -37.5025*tf.T913
                  + 4.4781*tf.T9 + -0.401587*tf.T953 + 8.35133*tf.lnT9)
    # wiesr
    rate += np.exp(  208.967 + -95.0429*tf.T9i + 381.949*tf.T913i + -608.493*tf.T913
                  + 34.7229*tf.T9 + -2.02465*tf.T953 + 297.922*tf.lnT9)

    rate_eval.p_c11__he4_b8 = rate

@numba.njit()
def he4_c11__n_o14(rate_eval, tf):
    # c11 + he4 --> n + o14
    rate = 0.0

    # rpsmr
    rate += np.exp(  15.1629 + -34.8289*tf.T9i + -1.31632*tf.T913i + 2.06431*tf.T913
                  + 0.0585225*tf.T9 + -0.00948426*tf.T953 + -1.11933*tf.lnT9)

    rate_eval.he4_c11__n_o14 = rate

@numba.njit()
def he4_c11__p_n14(rate_eval, tf):
    # c11 + he4 --> p + n14
    rate = 0.0

    # cf88n
    rate += np.exp(  36.613 + -31.883*tf.T913i + -0.361593*tf.T913
                  + -0.394216*tf.T9 + 0.0239162*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_c11__p_n14 = rate

@numba.njit()
def n_c12__p_b12(rate_eval, tf):
    # c12 + n --> p + b12
    rate = 0.0

    #  wagn
    rate += np.exp(  27.8183 + -146.08*tf.T9i + -12.12*tf.T913i + -1.83547e-09*tf.T913
                  + 2.0104e-10*tf.T9 + -1.82499e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_c12__p_b12 = rate

@numba.njit()
def n_c12__he4_be9(rate_eval, tf):
    # c12 + n --> he4 + be9
    rate = 0.0

    # cf88r
    rate += np.exp(  -7.17852 + -67.3413*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  33.8051 + -66.1573*tf.T9i + -23.87*tf.T913i + 0.566698*tf.T913
                  + 44.0957*tf.T9 + -314.232*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  21.6373 + -78.8893*tf.T9i)
    # cf88r
    rate += np.exp(  14.0851 + -70.3363*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88r
    rate += np.exp(  0.858256 + -67.9913*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.n_c12__he4_be9 = rate

@numba.njit()
def he4_c12__n_o15(rate_eval, tf):
    # c12 + he4 --> n + o15
    rate = 0.0

    # cf88n
    rate += np.exp(  17.0115 + -98.6615*tf.T9i + 0.124787*tf.T913
                  + 0.0588937*tf.T9 + -0.00679206*tf.T953)

    rate_eval.he4_c12__n_o15 = rate

@numba.njit()
def he4_c12__p_n15(rate_eval, tf):
    # c12 + he4 --> p + n15
    rate = 0.0

    # nacrn
    rate += np.exp(  27.118 + -57.6279*tf.T9i + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.93365 + -58.7917*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.5388 + -65.034*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -5.2319 + -59.6491*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.he4_c12__p_n15 = rate

@numba.njit()
def p_c13__n_n13(rate_eval, tf):
    # c13 + p --> n + n13
    rate = 0.0

    # nacrn
    rate += np.exp(  17.7625 + -34.8483*tf.T9i + 1.26126*tf.T913
                  + -0.204952*tf.T9 + 0.0310523*tf.T953)

    rate_eval.p_c13__n_n13 = rate

@numba.njit()
def p_c13__he4_b10(rate_eval, tf):
    # c13 + p --> he4 + b10
    rate = 0.0

    #  wagn
    rate += np.exp(  36.7435 + -47.1362*tf.T9i + -27.99*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.p_c13__he4_b10 = rate

@numba.njit()
def d_c13__n_n14(rate_eval, tf):
    # c13 + d --> n + n14
    rate = 0.0

    # bb92n
    rate += np.exp(  27.1993 + -0.00261944*tf.T9i + -16.8935*tf.T913i + 4.06445*tf.T913
                  + -1.1715*tf.T9 + 0.118556*tf.T953 + -1.13937*tf.lnT9)

    rate_eval.d_c13__n_n14 = rate

@numba.njit()
def he4_c13__n_o16(rate_eval, tf):
    # c13 + he4 --> n + o16
    rate = 0.0

    # gl12 
    rate += np.exp(  79.3008 + -0.30489*tf.T9i + 7.43132*tf.T913i + -84.8689*tf.T913
                  + 3.65083*tf.T9 + -0.148015*tf.T953 + 37.6008*tf.lnT9)
    # gl12 
    rate += np.exp(  62.5775 + -0.0277331*tf.T9i + -32.3917*tf.T913i + -48.934*tf.T913
                  + 44.1843*tf.T9 + -20.8743*tf.T953 + 2.02494*tf.lnT9)

    rate_eval.he4_c13__n_o16 = rate

@numba.njit()
def p_c14__n_n14(rate_eval, tf):
    # c14 + p --> n + n14
    rate = 0.0

    # cf88 
    rate += np.exp(  5.23589 + -7.26442*tf.T9i + 12.3428*tf.T913
                  + -2.70025*tf.T9 + 0.236625*tf.T953 + 1.0*tf.lnT9)
    # cf88n
    rate += np.exp(  14.7608 + -7.26442*tf.T9i + -4.33989*tf.T913
                  + 11.4311*tf.T9 + -11.7764*tf.T953)

    rate_eval.p_c14__n_n14 = rate

@numba.njit()
def p_c14__he4_b11(rate_eval, tf):
    # c14 + p --> he4 + b11
    rate = 0.0

    # bb92r
    rate += np.exp(  17.8245 + -20.357*tf.T9i + -1.87598e-08*tf.T913i + 3.26423e-08*tf.T913
                  + -2.18782e-09*tf.T9 + 1.43323e-10*tf.T953 + 0.6*tf.lnT9)
    # bb92r
    rate += np.exp(  -2.80313 + -11.965*tf.T9i + 2.62625e-09*tf.T913i + -6.58921e-09*tf.T913
                  + 5.62244e-10*tf.T9 + -4.28925e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  7.89937 + -14.244*tf.T9i + -5.81643e-09*tf.T913i + 1.24374e-08*tf.T913
                  + -9.55069e-10*tf.T9 + 6.81706e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  8.60505 + -14.254*tf.T9i + -2.8504e-09*tf.T913i + 5.85721e-09*tf.T913
                  + -4.34052e-10*tf.T9 + 3.01373e-11*tf.T953 + -1.5*tf.lnT9)
    # bb92n
    rate += np.exp(  180.727 + -9.29223*tf.T9i + 4.28912*tf.T913i + -214.72*tf.T913
                  + 57.4073*tf.T9 + -25.5329*tf.T953 + 53.0473*tf.lnT9)

    rate_eval.p_c14__he4_b11 = rate

@numba.njit()
def d_c14__n_n15(rate_eval, tf):
    # c14 + d --> n + n15
    rate = 0.0

    # bk92 
    rate += np.exp(  30.6841 + -16.939*tf.T913i + -0.582342*tf.T913
                  + -8.17066*tf.T9 + 1.70865*tf.T953 + -0.666667*tf.lnT9)
    # bk92 
    rate += np.exp(  33.5637 + -16.939*tf.T913i + -4.14392*tf.T913
                  + 0.438623*tf.T9 + -0.0354193*tf.T953 + 0.333333*tf.lnT9)

    rate_eval.d_c14__n_n15 = rate

@numba.njit()
def he4_n12__p_o15(rate_eval, tf):
    # n12 + he4 --> p + o15
    rate = 0.0

    #  wfhn
    rate += np.exp(  39.3263 + -35.6*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.he4_n12__p_o15 = rate

@numba.njit()
def n_n13__p_c13(rate_eval, tf):
    # n13 + n --> p + c13
    rate = 0.0

    # nacrn
    rate += np.exp(  17.7625 + 1.26126*tf.T913
                  + -0.204952*tf.T9 + 0.0310523*tf.T953)

    rate_eval.n_n13__p_c13 = rate

@numba.njit()
def n_n13__he4_b10(rate_eval, tf):
    # n13 + n --> he4 + b10
    rate = 0.0

    # cf88n
    rate += np.exp(  32.7497 + -12.2892*tf.T9i + -27.8719*tf.T913i + -0.599503*tf.T913
                  + 0.122849*tf.T9 + -0.0393717*tf.T953 + -0.507333*tf.lnT9)

    rate_eval.n_n13__he4_b10 = rate

@numba.njit()
def he4_n13__p_o16(rate_eval, tf):
    # n13 + he4 --> p + o16
    rate = 0.0

    # cf88n
    rate += np.exp(  40.4644 + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_n13__p_o16 = rate

@numba.njit()
def n_n14__p_c14(rate_eval, tf):
    # n14 + n --> p + c14
    rate = 0.0

    # cf88n
    rate += np.exp(  13.6622 + -4.33989*tf.T913
                  + 11.4311*tf.T9 + -11.7764*tf.T953)
    # cf88 
    rate += np.exp(  4.13728 + 12.3428*tf.T913
                  + -2.70025*tf.T9 + 0.236625*tf.T953 + 1.0*tf.lnT9)

    rate_eval.n_n14__p_c14 = rate

@numba.njit()
def n_n14__d_c13(rate_eval, tf):
    # n14 + n --> d + c13
    rate = 0.0

    # bb92n
    rate += np.exp(  28.1279 + -61.8182*tf.T9i + -16.8935*tf.T913i + 4.06445*tf.T913
                  + -1.1715*tf.T9 + 0.118556*tf.T953 + -1.13937*tf.lnT9)

    rate_eval.n_n14__d_c13 = rate

@numba.njit()
def n_n14__he4_b11(rate_eval, tf):
    # n14 + n --> he4 + b11
    rate = 0.0

    # cf88r
    rate += np.exp(  1.89445 + -4.66051*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  30.8848 + -1.83351*tf.T9i + -28.234*tf.T913i + -0.325987*tf.T913
                  + 30.135*tf.T9 + -78.4165*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  16.6206 + -10.4295*tf.T9i
                  + 0.6*tf.lnT9)
    # cf88r
    rate += np.exp(  8.75648 + -7.01151*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.n_n14__he4_b11 = rate

@numba.njit()
def p_n14__n_o14(rate_eval, tf):
    # n14 + p --> n + o14
    rate = 0.0

    # nacr 
    rate += np.exp(  11.3432 + -68.7567*tf.T9i + 5.48024*tf.T913
                  + -0.764072*tf.T9 + 0.0587804*tf.T953)

    rate_eval.p_n14__n_o14 = rate

@numba.njit()
def p_n14__he4_c11(rate_eval, tf):
    # n14 + p --> he4 + c11
    rate = 0.0

    # cf88n
    rate += np.exp(  37.9252 + -33.92*tf.T9i + -31.883*tf.T913i + -0.361593*tf.T913
                  + -0.394216*tf.T9 + 0.0239162*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_n14__he4_c11 = rate

@numba.njit()
def n_n15__d_c14(rate_eval, tf):
    # n15 + n --> d + c14
    rate = 0.0

    # bk92 
    rate += np.exp(  34.2122 + -92.6344*tf.T9i + -16.939*tf.T913i + -4.14392*tf.T913
                  + 0.438623*tf.T9 + -0.0354193*tf.T953 + 0.333333*tf.lnT9)
    # bk92 
    rate += np.exp(  31.3326 + -92.6344*tf.T9i + -16.939*tf.T913i + -0.582342*tf.T913
                  + -8.17066*tf.T9 + 1.70865*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_n15__d_c14 = rate

@numba.njit()
def n_n15__he4_b12(rate_eval, tf):
    # n15 + n --> he4 + b12
    rate = 0.0

    #  wagn
    rate += np.exp(  37.1076 + -88.4443*tf.T9i + -28.45*tf.T913i + 2.82044e-10*tf.T913
                  + -2.59013e-11*tf.T9 + 2.07893e-12*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_n15__he4_b12 = rate

@numba.njit()
def p_n15__n_o15(rate_eval, tf):
    # n15 + p --> n + o15
    rate = 0.0

    # nacrn
    rate += np.exp(  18.3942 + -41.0335*tf.T9i + 0.331392*tf.T913
                  + 0.0171473*tf.T9)

    rate_eval.p_n15__n_o15 = rate

@numba.njit()
def p_n15__he4_c12(rate_eval, tf):
    # n15 + p --> he4 + c12
    rate = 0.0

    # nacrn
    rate += np.exp(  27.4764 + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.57522 + -1.1638*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.8972 + -7.406*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -4.87347 + -2.02117*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_n15__he4_c12 = rate

@numba.njit()
def n_o14__p_n14(rate_eval, tf):
    # o14 + n --> p + n14
    rate = 0.0

    # nacr 
    rate += np.exp(  12.4418 + 5.48024*tf.T913
                  + -0.764072*tf.T9 + 0.0587804*tf.T953)

    rate_eval.n_o14__p_n14 = rate

@numba.njit()
def n_o14__he4_c11(rate_eval, tf):
    # o14 + n --> he4 + c11
    rate = 0.0

    # rpsmr
    rate += np.exp(  17.5737 + 0.00781138*tf.T9i + -1.31632*tf.T913i + 2.06431*tf.T913
                  + 0.0585225*tf.T9 + -0.00948426*tf.T953 + -1.11933*tf.lnT9)

    rate_eval.n_o14__he4_c11 = rate

@numba.njit()
def n_o15__p_n15(rate_eval, tf):
    # o15 + n --> p + n15
    rate = 0.0

    # nacrn
    rate += np.exp(  18.3942 + 0.331392*tf.T913
                  + 0.0171473*tf.T9)

    rate_eval.n_o15__p_n15 = rate

@numba.njit()
def n_o15__he4_c12(rate_eval, tf):
    # o15 + n --> he4 + c12
    rate = 0.0

    # cf88n
    rate += np.exp(  17.3699 + 0.124787*tf.T913
                  + 0.0588937*tf.T9 + -0.00679206*tf.T953)

    rate_eval.n_o15__he4_c12 = rate

@numba.njit()
def p_o15__he4_n12(rate_eval, tf):
    # o15 + p --> he4 + n12
    rate = 0.0

    #  wfhn
    rate += np.exp(  40.7833 + -111.611*tf.T9i + -35.6*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.p_o15__he4_n12 = rate

@numba.njit()
def n_o16__he4_c13(rate_eval, tf):
    # o16 + n --> he4 + c13
    rate = 0.0

    # gl12 
    rate += np.exp(  81.0688 + -26.0159*tf.T9i + 7.43132*tf.T913i + -84.8689*tf.T913
                  + 3.65083*tf.T9 + -0.148015*tf.T953 + 37.6008*tf.lnT9)
    # gl12 
    rate += np.exp(  64.3455 + -25.7388*tf.T9i + -32.3917*tf.T913i + -48.934*tf.T913
                  + 44.1843*tf.T9 + -20.8743*tf.T953 + 2.02494*tf.lnT9)

    rate_eval.n_o16__he4_c13 = rate

@numba.njit()
def p_o16__he4_n13(rate_eval, tf):
    # o16 + p --> he4 + n13
    rate = 0.0

    # cf88n
    rate += np.exp(  42.2324 + -60.5523*tf.T9i + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_o16__he4_n13 = rate

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
def p_li8__n_he4_he4(rate_eval, tf):
    # li8 + p --> n + he4 + he4
    rate = 0.0

    # bb92r
    rate += np.exp(  19.6085 + -7.024*tf.T9i + -1.51325e-10*tf.T913i + 4.9858e-10*tf.T913
                  + -4.87305e-11*tf.T9 + 4.0038e-12*tf.T953 + -1.5*tf.lnT9)
    # bb92r
    rate += np.exp(  20.8455 + -3.982*tf.T9i + -3.90706e-09*tf.T913i + 9.14745e-09*tf.T913
                  + -7.44527e-10*tf.T9 + 5.50866e-11*tf.T953 + -0.433*tf.lnT9)
    # bb92n
    rate += np.exp(  23.0564 + -4.7809e-12*tf.T9i + -8.429*tf.T913i + -2.04701e-09*tf.T913
                  + 2.04027e-10*tf.T9 + -1.70107e-11*tf.T953 + -0.666667*tf.lnT9)
    # bb92r
    rate += np.exp(  13.4284 + -1.02*tf.T9i + -1.56584e-10*tf.T913i + 4.97791e-10*tf.T913
                  + -4.88505e-11*tf.T9 + 4.04166e-12*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_li8__n_he4_he4 = rate

@numba.njit()
def d_be7__p_he4_he4(rate_eval, tf):
    # be7 + d --> p + he4 + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  27.6987 + -12.428*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_be7__p_he4_he4 = rate

@numba.njit()
def p_be9__d_he4_he4(rate_eval, tf):
    # be9 + p --> d + he4 + he4
    rate = 0.0

    # cf88r
    rate += np.exp(  20.5607 + -5.8*tf.T9i
                  + -0.75*tf.lnT9)
    # cf88r
    rate += np.exp(  20.1768 + -3.046*tf.T9i
                  + -1.0*tf.lnT9)
    # cf88n
    rate += np.exp(  26.0751 + -10.359*tf.T913i + 0.103955*tf.T913
                  + 4.4262*tf.T9 + -5.95664*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_be9__d_he4_he4 = rate

@numba.njit()
def n_b8__p_he4_he4(rate_eval, tf):
    # b8 + n --> p + he4 + he4
    rate = 0.0

    #  wagn
    rate += np.exp(  19.812 + -1.13869e-12*tf.T9i + 1.5368e-10*tf.T913i + -5.36043e-10*tf.T913
                  + 5.80628e-11*tf.T9 + -5.23778e-12*tf.T953 + 1.77211e-10*tf.lnT9)

    rate_eval.n_b8__p_he4_he4 = rate

@numba.njit()
def p_b11__he4_he4_he4(rate_eval, tf):
    # b11 + p --> he4 + he4 + he4
    rate = 0.0

    # nacrr
    rate += np.exp(  -14.9395 + -1.724*tf.T9i + 8.49175*tf.T913i + 27.3254*tf.T913
                  + -3.72071*tf.T9 + 0.275516*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  28.6442 + -12.097*tf.T913i + -0.0496312*tf.T913
                  + 0.687736*tf.T9 + -0.564229*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_b11__he4_he4_he4 = rate

@numba.njit()
def n_c11__he4_he4_he4(rate_eval, tf):
    # c11 + n --> he4 + he4 + he4
    rate = 0.0

    # bb92r
    rate += np.exp(  22.7448 + -5.58*tf.T9i
                  + -1.5*tf.lnT9)
    # bb92n
    rate += np.exp(  18.9275)
    # bb92r
    rate += np.exp(  14.7197 + -0.917*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.n_c11__he4_he4_he4 = rate

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
def p_be9__n_p_he4_he4(rate_eval, tf):
    # be9 + p --> n + p + he4 + he4
    rate = 0.0

    # cf88r
    rate += np.exp(  20.7431 + -26.725*tf.T9i + 1.40505e-06*tf.T913i + -1.47128e-06*tf.T913
                  + 6.89313e-08*tf.T9 + -3.55179e-09*tf.T953 + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  14.6035 + -21.4742*tf.T9i + -0.634849*tf.T913i + 4.82033*tf.T913
                  + -0.257317*tf.T9 + 0.0134206*tf.T953 + -1.08885*tf.lnT9)

    rate_eval.p_be9__n_p_he4_he4 = rate

@numba.njit()
def n_p_he4__li6(rate_eval, tf):
    # n + p + he4 --> li6
    rate = 0.0

    # cf88r
    rate += np.exp(  -12.2851 + -19.353*tf.T9i + 1.44987*tf.T913i + -1.42759*tf.T913
                  + 0.0454035*tf.T9 + 0.00471161*tf.T953 + -1.0*tf.lnT9)

    rate_eval.n_p_he4__li6 = rate

@numba.njit()
def n_he4_he4__be9(rate_eval, tf):
    # n + he4 + he4 --> be9
    rate = 0.0

    # ac12r
    rate += np.exp(  -6.81178 + -1.01953*tf.T9i + -1.56673*tf.T913i + -5.43497*tf.T913
                  + 0.673807*tf.T9 + -0.041014*tf.T953 + -1.5*tf.lnT9)
    # ac12n
    rate += np.exp(  -8.22898 + -13.3317*tf.T913i + 13.2237*tf.T913
                  + -9.06339*tf.T9 + -0.666667*tf.lnT9)

    rate_eval.n_he4_he4__be9 = rate

@numba.njit()
def he4_he4_he4__c12(rate_eval, tf):
    # he4 + he4 + he4 --> c12
    rate = 0.0

    # fy05r
    rate += np.exp(  -24.3505 + -4.12656*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -13.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  -11.7884 + -1.02446*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + -2.16667*tf.lnT9)
    # fy05n
    rate += np.exp(  -0.971052 + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + -1.33333*tf.lnT9)

    rate_eval.he4_he4_he4__c12 = rate

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
def n_he4_he4__p_li8(rate_eval, tf):
    # n + he4 + he4 --> p + li8
    rate = 0.0

    # bb92r
    rate += np.exp(  -0.904176 + -181.687*tf.T9i + -3.90706e-09*tf.T913i + 9.14745e-09*tf.T913
                  + -7.44527e-10*tf.T9 + 5.50866e-11*tf.T953 + -1.933*tf.lnT9)
    # bb92n
    rate += np.exp(  1.30672 + -177.705*tf.T9i + -8.429*tf.T913i + -2.04701e-09*tf.T913
                  + 2.04027e-10*tf.T9 + -1.70107e-11*tf.T953 + -2.16667*tf.lnT9)
    # bb92r
    rate += np.exp(  -8.32128 + -178.725*tf.T9i + -1.56584e-10*tf.T913i + 4.97791e-10*tf.T913
                  + -4.88505e-11*tf.T9 + 4.04166e-12*tf.T953 + -3.0*tf.lnT9)
    # bb92r
    rate += np.exp(  -2.14118 + -184.729*tf.T9i + -1.51325e-10*tf.T913i + 4.9858e-10*tf.T913
                  + -4.87305e-11*tf.T9 + 4.0038e-12*tf.T953 + -3.0*tf.lnT9)

    rate_eval.n_he4_he4__p_li8 = rate

@numba.njit()
def n_he4_he4__d_li7(rate_eval, tf):
    # n + he4 + he4 --> d + li7
    rate = 0.0

    # cf88n
    rate += np.exp(  5.67199 + -175.472*tf.T9i + -10.259*tf.T913i
                  + -2.16667*tf.lnT9)

    rate_eval.n_he4_he4__d_li7 = rate

@numba.njit()
def p_he4_he4__n_b8(rate_eval, tf):
    # p + he4 + he4 --> n + b8
    rate = 0.0

    #  wagn
    rate += np.exp(  -1.93853 + -218.783*tf.T9i + 1.5368e-10*tf.T913i + -5.36043e-10*tf.T913
                  + 5.80628e-11*tf.T9 + -5.23778e-12*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_he4_he4__n_b8 = rate

@numba.njit()
def p_he4_he4__d_be7(rate_eval, tf):
    # p + he4 + he4 --> d + be7
    rate = 0.0

    # cf88n
    rate += np.exp(  6.97069 + -194.561*tf.T9i + -12.428*tf.T913i
                  + -2.16667*tf.lnT9)

    rate_eval.p_he4_he4__d_be7 = rate

@numba.njit()
def d_he4_he4__p_be9(rate_eval, tf):
    # d + he4 + he4 --> p + be9
    rate = 0.0

    # cf88r
    rate += np.exp(  -2.68071 + -13.3545*tf.T9i
                  + -2.25*tf.lnT9)
    # cf88r
    rate += np.exp(  -3.06461 + -10.6005*tf.T9i
                  + -2.5*tf.lnT9)
    # cf88n
    rate += np.exp(  2.83369 + -7.55453*tf.T9i + -10.359*tf.T913i + 0.103955*tf.T913
                  + 4.4262*tf.T9 + -5.95664*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.d_he4_he4__p_be9 = rate

@numba.njit()
def he4_he4_he4__n_c11(rate_eval, tf):
    # he4 + he4 + he4 --> n + c11
    rate = 0.0

    # bb92r
    rate += np.exp(  -7.0632 + -133.749*tf.T9i
                  + -3.0*tf.lnT9)
    # bb92r
    rate += np.exp(  0.961896 + -138.412*tf.T9i
                  + -3.0*tf.lnT9)
    # bb92n
    rate += np.exp(  -2.8554 + -132.832*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_he4_he4__n_c11 = rate

@numba.njit()
def he4_he4_he4__p_b11(rate_eval, tf):
    # he4 + he4 + he4 --> p + b11
    rate = 0.0

    # nacrr
    rate += np.exp(  -36.7224 + -102.474*tf.T9i + 8.49175*tf.T913i + 27.3254*tf.T913
                  + -3.72071*tf.T9 + 0.275516*tf.T953 + -3.0*tf.lnT9)
    # nacrn
    rate += np.exp(  6.8613 + -100.75*tf.T9i + -12.097*tf.T913i + -0.0496312*tf.T913
                  + 0.687736*tf.T9 + -0.564229*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.he4_he4_he4__p_b11 = rate

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
def n_p_he4_he4__p_be9(rate_eval, tf):
    # n + p + he4 + he4 --> p + be9
    rate = 0.0

    # cf88r
    rate += np.exp(  -25.452 + -8.47112*tf.T9i + 1.40505e-06*tf.T913i + -1.47128e-06*tf.T913
                  + 6.89313e-08*tf.T9 + -3.55179e-09*tf.T953 + -4.5*tf.lnT9)
    # cf88n
    rate += np.exp(  -31.5916 + -3.22032*tf.T9i + -0.634849*tf.T913i + 4.82033*tf.T913
                  + -0.257317*tf.T9 + 0.0134206*tf.T953 + -4.08885*tf.lnT9)

    rate_eval.n_p_he4_he4__p_be9 = rate

@numba.njit()
def p_p_he4_he4__he3_be7(rate_eval, tf):
    # p + p + he4 + he4 --> he3 + be7
    rate = 0.0

    # mafon
    rate += np.exp(  -13.1807 + -130.807*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -3.66667*tf.lnT9)

    rate_eval.p_p_he4_he4__he3_be7 = rate


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
    be7__li7__weak__electron_capture(rate_eval, tf)
    b12__c12__weak__wc17(rate_eval, tf)
    c11__b11__weak__wc12(rate_eval, tf)
    c14__n14__weak__wc12(rate_eval, tf)
    n12__c12__weak__wc12(rate_eval, tf)
    n13__c13__weak__wc12(rate_eval, tf)
    o14__n14__weak__wc12(rate_eval, tf)
    o15__n15__weak__wc12(rate_eval, tf)
    d__n_p(rate_eval, tf)
    t__n_d(rate_eval, tf)
    he3__p_d(rate_eval, tf)
    he4__n_he3(rate_eval, tf)
    he4__p_t(rate_eval, tf)
    he4__d_d(rate_eval, tf)
    li6__he4_d(rate_eval, tf)
    li7__n_li6(rate_eval, tf)
    li7__he4_t(rate_eval, tf)
    li8__n_li7(rate_eval, tf)
    li8__he4_he4__weak__wc12(rate_eval, tf)
    be7__p_li6(rate_eval, tf)
    be7__he4_he3(rate_eval, tf)
    b8__p_be7(rate_eval, tf)
    b8__he4_he4__weak__wc12(rate_eval, tf)
    b10__p_be9(rate_eval, tf)
    b10__he4_li6(rate_eval, tf)
    b11__n_b10(rate_eval, tf)
    b11__he4_li7(rate_eval, tf)
    b12__n_b11(rate_eval, tf)
    b12__he4_li8(rate_eval, tf)
    c11__p_b10(rate_eval, tf)
    c11__he4_be7(rate_eval, tf)
    c12__n_c11(rate_eval, tf)
    c12__p_b11(rate_eval, tf)
    c13__n_c12(rate_eval, tf)
    c14__n_c13(rate_eval, tf)
    n12__p_c11(rate_eval, tf)
    n13__p_c12(rate_eval, tf)
    n14__n_n13(rate_eval, tf)
    n14__p_c13(rate_eval, tf)
    n15__n_n14(rate_eval, tf)
    n15__p_c14(rate_eval, tf)
    o14__p_n13(rate_eval, tf)
    o15__n_o14(rate_eval, tf)
    o15__p_n14(rate_eval, tf)
    o16__n_o15(rate_eval, tf)
    o16__p_n15(rate_eval, tf)
    o16__he4_c12(rate_eval, tf)
    li6__n_p_he4(rate_eval, tf)
    be9__n_he4_he4(rate_eval, tf)
    c12__he4_he4_he4(rate_eval, tf)
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
    he4_li6__b10(rate_eval, tf)
    n_li7__li8(rate_eval, tf)
    he4_li7__b11(rate_eval, tf)
    he4_li8__b12(rate_eval, tf)
    p_be7__b8(rate_eval, tf)
    he4_be7__c11(rate_eval, tf)
    p_be9__b10(rate_eval, tf)
    n_b10__b11(rate_eval, tf)
    p_b10__c11(rate_eval, tf)
    n_b11__b12(rate_eval, tf)
    p_b11__c12(rate_eval, tf)
    n_c11__c12(rate_eval, tf)
    p_c11__n12(rate_eval, tf)
    n_c12__c13(rate_eval, tf)
    p_c12__n13(rate_eval, tf)
    he4_c12__o16(rate_eval, tf)
    n_c13__c14(rate_eval, tf)
    p_c13__n14(rate_eval, tf)
    p_c14__n15(rate_eval, tf)
    n_n13__n14(rate_eval, tf)
    p_n13__o14(rate_eval, tf)
    n_n14__n15(rate_eval, tf)
    p_n14__o15(rate_eval, tf)
    p_n15__o16(rate_eval, tf)
    n_o14__o15(rate_eval, tf)
    n_o15__o16(rate_eval, tf)
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
    he4_li6__p_be9(rate_eval, tf)
    p_li7__n_be7(rate_eval, tf)
    p_li7__d_li6(rate_eval, tf)
    p_li7__he4_he4(rate_eval, tf)
    d_li7__p_li8(rate_eval, tf)
    t_li7__n_be9(rate_eval, tf)
    t_li7__d_li8(rate_eval, tf)
    he4_li7__n_b10(rate_eval, tf)
    p_li8__d_li7(rate_eval, tf)
    d_li8__n_be9(rate_eval, tf)
    d_li8__t_li7(rate_eval, tf)
    he4_li8__n_b11(rate_eval, tf)
    n_be7__p_li7(rate_eval, tf)
    n_be7__d_li6(rate_eval, tf)
    n_be7__he4_he4(rate_eval, tf)
    he4_be7__p_b10(rate_eval, tf)
    n_be9__d_li8(rate_eval, tf)
    n_be9__t_li7(rate_eval, tf)
    p_be9__he4_li6(rate_eval, tf)
    t_be9__n_b11(rate_eval, tf)
    he4_be9__n_c12(rate_eval, tf)
    he4_be9__p_b12(rate_eval, tf)
    he4_b8__p_c11(rate_eval, tf)
    n_b10__he4_li7(rate_eval, tf)
    p_b10__he4_be7(rate_eval, tf)
    he4_b10__n_n13(rate_eval, tf)
    he4_b10__p_c13(rate_eval, tf)
    n_b11__t_be9(rate_eval, tf)
    n_b11__he4_li8(rate_eval, tf)
    p_b11__n_c11(rate_eval, tf)
    he4_b11__n_n14(rate_eval, tf)
    he4_b11__p_c14(rate_eval, tf)
    p_b12__n_c12(rate_eval, tf)
    p_b12__he4_be9(rate_eval, tf)
    he4_b12__n_n15(rate_eval, tf)
    n_c11__p_b11(rate_eval, tf)
    p_c11__he4_b8(rate_eval, tf)
    he4_c11__n_o14(rate_eval, tf)
    he4_c11__p_n14(rate_eval, tf)
    n_c12__p_b12(rate_eval, tf)
    n_c12__he4_be9(rate_eval, tf)
    he4_c12__n_o15(rate_eval, tf)
    he4_c12__p_n15(rate_eval, tf)
    p_c13__n_n13(rate_eval, tf)
    p_c13__he4_b10(rate_eval, tf)
    d_c13__n_n14(rate_eval, tf)
    he4_c13__n_o16(rate_eval, tf)
    p_c14__n_n14(rate_eval, tf)
    p_c14__he4_b11(rate_eval, tf)
    d_c14__n_n15(rate_eval, tf)
    he4_n12__p_o15(rate_eval, tf)
    n_n13__p_c13(rate_eval, tf)
    n_n13__he4_b10(rate_eval, tf)
    he4_n13__p_o16(rate_eval, tf)
    n_n14__p_c14(rate_eval, tf)
    n_n14__d_c13(rate_eval, tf)
    n_n14__he4_b11(rate_eval, tf)
    p_n14__n_o14(rate_eval, tf)
    p_n14__he4_c11(rate_eval, tf)
    n_n15__d_c14(rate_eval, tf)
    n_n15__he4_b12(rate_eval, tf)
    p_n15__n_o15(rate_eval, tf)
    p_n15__he4_c12(rate_eval, tf)
    n_o14__p_n14(rate_eval, tf)
    n_o14__he4_c11(rate_eval, tf)
    n_o15__p_n15(rate_eval, tf)
    n_o15__he4_c12(rate_eval, tf)
    p_o15__he4_n12(rate_eval, tf)
    n_o16__he4_c13(rate_eval, tf)
    p_o16__he4_n13(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    t_t__n_n_he4(rate_eval, tf)
    t_he3__n_p_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    d_li7__n_he4_he4(rate_eval, tf)
    p_li8__n_he4_he4(rate_eval, tf)
    d_be7__p_he4_he4(rate_eval, tf)
    p_be9__d_he4_he4(rate_eval, tf)
    n_b8__p_he4_he4(rate_eval, tf)
    p_b11__he4_he4_he4(rate_eval, tf)
    n_c11__he4_he4_he4(rate_eval, tf)
    t_li7__n_n_he4_he4(rate_eval, tf)
    he3_li7__n_p_he4_he4(rate_eval, tf)
    t_be7__n_p_he4_he4(rate_eval, tf)
    he3_be7__p_p_he4_he4(rate_eval, tf)
    p_be9__n_p_he4_he4(rate_eval, tf)
    n_p_he4__li6(rate_eval, tf)
    n_he4_he4__be9(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)
    n_n_he4__t_t(rate_eval, tf)
    n_p_he4__t_he3(rate_eval, tf)
    p_p_he4__he3_he3(rate_eval, tf)
    n_he4_he4__p_li8(rate_eval, tf)
    n_he4_he4__d_li7(rate_eval, tf)
    p_he4_he4__n_b8(rate_eval, tf)
    p_he4_he4__d_be7(rate_eval, tf)
    d_he4_he4__p_be9(rate_eval, tf)
    he4_he4_he4__n_c11(rate_eval, tf)
    he4_he4_he4__p_b11(rate_eval, tf)
    n_n_he4_he4__t_li7(rate_eval, tf)
    n_p_he4_he4__he3_li7(rate_eval, tf)
    n_p_he4_he4__t_be7(rate_eval, tf)
    n_p_he4_he4__p_be9(rate_eval, tf)
    p_p_he4_he4__he3_be7(rate_eval, tf)

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

        scn_fac = ScreenFactors(2, 4, 3, 6)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li6__b10 *= scor
        rate_eval.he4_li6__p_be9 *= scor

        scn_fac = ScreenFactors(2, 4, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li7__b11 *= scor
        rate_eval.he4_li7__n_b10 *= scor

        scn_fac = ScreenFactors(2, 4, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li8__b12 *= scor
        rate_eval.he4_li8__n_b11 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be7__b8 *= scor

        scn_fac = ScreenFactors(2, 4, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_be7__c11 *= scor
        rate_eval.he4_be7__p_b10 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be9__b10 *= scor
        rate_eval.p_be9__he4_li6 *= scor
        rate_eval.p_be9__d_he4_he4 *= scor
        rate_eval.p_be9__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 10)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b10__c11 *= scor
        rate_eval.p_b10__he4_be7 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b11__c12 *= scor
        rate_eval.p_b11__n_c11 *= scor
        rate_eval.p_b11__he4_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c11__n12 *= scor
        rate_eval.p_c11__he4_b8 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c12__n13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c12__o16 *= scor
        rate_eval.he4_c12__n_o15 *= scor
        rate_eval.he4_c12__p_n15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c13__n14 *= scor
        rate_eval.p_c13__n_n13 *= scor
        rate_eval.p_c13__he4_b10 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c14__n15 *= scor
        rate_eval.p_c14__n_n14 *= scor
        rate_eval.p_c14__he4_b11 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n13__o14 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n14__o15 *= scor
        rate_eval.p_n14__n_o14 *= scor
        rate_eval.p_n14__he4_c11 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n15__o16 *= scor
        rate_eval.p_n15__n_o15 *= scor
        rate_eval.p_n15__he4_c12 *= scor

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
        rate_eval.n_he4_he4__be9 *= scor
        rate_eval.n_he4_he4__p_li8 *= scor
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

        scn_fac = ScreenFactors(1, 2, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li7__p_li8 *= scor
        rate_eval.d_li7__n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_li7__n_be9 *= scor
        rate_eval.t_li7__d_li8 *= scor
        rate_eval.t_li7__n_n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li8__d_li7 *= scor
        rate_eval.p_li8__n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li8__n_be9 *= scor
        rate_eval.d_li8__t_li7 *= scor

        scn_fac = ScreenFactors(1, 3, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_be9__n_b11 *= scor

        scn_fac = ScreenFactors(2, 4, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_be9__n_c12 *= scor
        rate_eval.he4_be9__p_b12 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b8__p_c11 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 10)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b10__n_n13 *= scor
        rate_eval.he4_b10__p_c13 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b11__n_n14 *= scor
        rate_eval.he4_b11__p_c14 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b12__n_c12 *= scor
        rate_eval.p_b12__he4_be9 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b12__n_n15 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c11__n_o14 *= scor
        rate_eval.he4_c11__p_n14 *= scor

        scn_fac = ScreenFactors(1, 2, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_c13__n_n14 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c13__n_o16 *= scor

        scn_fac = ScreenFactors(1, 2, 6, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_c14__n_n15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n12__p_o15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n13__p_o16 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o15__he4_n12 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o16__he4_n13 *= scor

        scn_fac = ScreenFactors(1, 3, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_t__n_n_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_be7__p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_li7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_be7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_be7__p_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2
        rate_eval.he4_he4_he4__n_c11 *= scor * scor2
        rate_eval.he4_he4_he4__p_b11 *= scor * scor2

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_he4__he3_he3 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he4_he4__n_b8 *= scor
        rate_eval.p_he4_he4__d_be7 *= scor
        rate_eval.n_p_he4_he4__he3_li7 *= scor
        rate_eval.n_p_he4_he4__t_be7 *= scor
        rate_eval.n_p_he4_he4__p_be9 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he4_he4__p_be9 *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -rho*Y[jn]*Y[jd]*rate_eval.n_d__t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jn]*Y[jli7]*rate_eval.n_li7__li8
       -rho*Y[jn]*Y[jb10]*rate_eval.n_b10__b11
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__b12
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__c12
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__c13
       -rho*Y[jn]*Y[jc13]*rate_eval.n_c13__c14
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__n14
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__n15
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__o15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__o16
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__he4_he4
       -rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__d_li8
       -rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__t_li7
       -rho*Y[jn]*Y[jb10]*rate_eval.n_b10__he4_li7
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__t_be9
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__he4_li8
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__p_b11
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__p_b12
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__he4_be9
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__p_c13
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__he4_b10
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__p_c14
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__d_c13
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__he4_b11
       -rho*Y[jn]*Y[jn15]*rate_eval.n_n15__d_c14
       -rho*Y[jn]*Y[jn15]*rate_eval.n_n15__he4_b12
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__p_n14
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__he4_c11
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__p_n15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__he4_c12
       -rho*Y[jn]*Y[jo16]*rate_eval.n_o16__he4_c13
       -rho*Y[jn]*Y[jb8]*rate_eval.n_b8__p_he4_he4
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -2*5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       -5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -Y[jn]*rate_eval.n__p
       +Y[jd]*rate_eval.d__n_p
       +Y[jt]*rate_eval.t__n_d
       +Y[jhe4]*rate_eval.he4__n_he3
       +Y[jli7]*rate_eval.li7__n_li6
       +Y[jli8]*rate_eval.li8__n_li7
       +Y[jb11]*rate_eval.b11__n_b10
       +Y[jb12]*rate_eval.b12__n_b11
       +Y[jc12]*rate_eval.c12__n_c11
       +Y[jc13]*rate_eval.c13__n_c12
       +Y[jc14]*rate_eval.c14__n_c13
       +Y[jn14]*rate_eval.n14__n_n13
       +Y[jn15]*rate_eval.n15__n_n14
       +Y[jo15]*rate_eval.o15__n_o14
       +Y[jo16]*rate_eval.o16__n_o15
       +Y[jli6]*rate_eval.li6__n_p_he4
       +Y[jbe9]*rate_eval.be9__n_he4_he4
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__n_he3
       +rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jd]*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       +rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_be9
       +rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__n_b10
       +rho*Y[jd]*Y[jli8]*rate_eval.d_li8__n_be9
       +rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__n_b11
       +rho*Y[jt]*Y[jbe9]*rate_eval.t_be9__n_b11
       +rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__n_c12
       +rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__n_n13
       +rho*Y[jp]*Y[jb11]*rate_eval.p_b11__n_c11
       +rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__n_n14
       +rho*Y[jp]*Y[jb12]*rate_eval.p_b12__n_c12
       +rho*Y[jhe4]*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__n_o14
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__n_o15
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n_n13
       +rho*Y[jd]*Y[jc13]*rate_eval.d_c13__n_n14
       +rho*Y[jhe4]*Y[jc13]*rate_eval.he4_c13__n_o16
       +rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n_n14
       +rho*Y[jd]*Y[jc14]*rate_eval.d_c14__n_n15
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__n_o14
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__n_o15
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +2*5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +rho*Y[jp]*Y[jli8]*rate_eval.p_li8__n_he4_he4
       +2*rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__n_c11
       +Y[jp]*rate_eval.p__n
       )

    dYdt[jp] = (
       -rho*Y[jn]*Y[jp]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__b10
       -rho*Y[jp]*Y[jb10]*rate_eval.p_b10__c11
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__c12
       -rho*Y[jp]*Y[jc11]*rate_eval.p_c11__n12
       -rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n15
       -rho*Y[jp]*Y[jn13]*rate_eval.p_n13__o14
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jp]*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       -rho*Y[jp]*Y[jli8]*rate_eval.p_li8__d_li7
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__he4_li6
       -rho*Y[jp]*Y[jb10]*rate_eval.p_b10__he4_be7
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__n_c11
       -rho*Y[jp]*Y[jb12]*rate_eval.p_b12__n_c12
       -rho*Y[jp]*Y[jb12]*rate_eval.p_b12__he4_be9
       -rho*Y[jp]*Y[jc11]*rate_eval.p_c11__he4_b8
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n_n13
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__he4_b10
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n_n14
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__he4_b11
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__n_o14
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__he4_c11
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__n_o15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       -rho*Y[jp]*Y[jo15]*rate_eval.p_o15__he4_n12
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -rho*Y[jp]*Y[jli8]*rate_eval.p_li8__n_he4_he4
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       -5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       -Y[jp]*rate_eval.p__n
       +Y[jd]*rate_eval.d__n_p
       +Y[jhe3]*rate_eval.he3__p_d
       +Y[jhe4]*rate_eval.he4__p_t
       +Y[jbe7]*rate_eval.be7__p_li6
       +Y[jb8]*rate_eval.b8__p_be7
       +Y[jb10]*rate_eval.b10__p_be9
       +Y[jc11]*rate_eval.c11__p_b10
       +Y[jc12]*rate_eval.c12__p_b11
       +Y[jn12]*rate_eval.n12__p_c11
       +Y[jn13]*rate_eval.n13__p_c12
       +Y[jn14]*rate_eval.n14__p_c13
       +Y[jn15]*rate_eval.n15__p_c14
       +Y[jo14]*rate_eval.o14__p_n13
       +Y[jo15]*rate_eval.o15__p_n14
       +Y[jo16]*rate_eval.o16__p_n15
       +Y[jli6]*rate_eval.li6__n_p_he4
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__p_t
       +rho*Y[jn]*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__p_be9
       +rho*Y[jd]*Y[jli7]*rate_eval.d_li7__p_li8
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       +rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__p_b10
       +rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__p_b12
       +rho*Y[jhe4]*Y[jb8]*rate_eval.he4_b8__p_c11
       +rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__p_c13
       +rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__p_c14
       +rho*Y[jn]*Y[jc11]*rate_eval.n_c11__p_b11
       +rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__p_n14
       +rho*Y[jn]*Y[jc12]*rate_eval.n_c12__p_b12
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jhe4]*Y[jn12]*rate_eval.he4_n12__p_o15
       +rho*Y[jn]*Y[jn13]*rate_eval.n_n13__p_c13
       +rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__p_c14
       +rho*Y[jn]*Y[jo14]*rate_eval.n_o14__p_n14
       +rho*Y[jn]*Y[jo15]*rate_eval.n_o15__p_n15
       +2*rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +rho*Y[jn]*Y[jb8]*rate_eval.n_b8__p_he4_he4
       +rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jp]**2*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       +5.00000000000000e-01*rho**2*Y[jd]*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__p_b11
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       +Y[jn]*rate_eval.n__p
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
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__p_li8
       -rho*Y[jd]*Y[jli8]*rate_eval.d_li8__n_be9
       -rho*Y[jd]*Y[jli8]*rate_eval.d_li8__t_li7
       -rho*Y[jd]*Y[jc13]*rate_eval.d_c13__n_n14
       -rho*Y[jd]*Y[jc14]*rate_eval.d_c14__n_n15
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__n_p_p
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       -5.00000000000000e-01*rho**2*Y[jd]*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
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
       +rho*Y[jt]*Y[jli7]*rate_eval.t_li7__d_li8
       +rho*Y[jp]*Y[jli8]*rate_eval.p_li8__d_li7
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       +rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__d_li8
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__d_c13
       +rho*Y[jn]*Y[jn15]*rate_eval.n_n15__d_c14
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__d_he4_he4
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
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_be9
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__d_li8
       -rho*Y[jt]*Y[jbe9]*rate_eval.t_be9__n_b11
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
       +rho*Y[jd]*Y[jli8]*rate_eval.d_li8__t_li7
       +rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__t_li7
       +rho*Y[jn]*Y[jb11]*rate_eval.n_b11__t_be9
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
       -rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__b10
       -rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__b11
       -rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__b12
       -rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__c11
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       -rho*Y[jn]*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jp]*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*Y[jhe4]*rate_eval.d_he4__t_he3
       -2*5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       -2*5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       -rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__p_be9
       -rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__n_b10
       -rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__n_b11
       -rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__p_b10
       -rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__n_c12
       -rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__p_b12
       -rho*Y[jhe4]*Y[jb8]*rate_eval.he4_b8__p_c11
       -rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__n_n13
       -rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__p_c13
       -rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__n_n14
       -rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__p_c14
       -rho*Y[jhe4]*Y[jb12]*rate_eval.he4_b12__n_n15
       -rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__n_o14
       -rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__p_n14
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__n_o15
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       -rho*Y[jhe4]*Y[jc13]*rate_eval.he4_c13__n_o16
       -rho*Y[jhe4]*Y[jn12]*rate_eval.he4_n12__p_o15
       -rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       -5.00000000000000e-01*rho**2*Y[jn]**2*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       -2*5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       -2*5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -2*5.00000000000000e-01*rho**2*Y[jd]*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__n_c11
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__p_b11
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       +Y[jli6]*rate_eval.li6__he4_d
       +Y[jli7]*rate_eval.li7__he4_t
       +2*Y[jli8]*rate_eval.li8__he4_he4__weak__wc12
       +Y[jbe7]*rate_eval.be7__he4_he3
       +2*Y[jb8]*rate_eval.b8__he4_he4__weak__wc12
       +Y[jb10]*rate_eval.b10__he4_li6
       +Y[jb11]*rate_eval.b11__he4_li7
       +Y[jb12]*rate_eval.b12__he4_li8
       +Y[jc11]*rate_eval.c11__he4_be7
       +Y[jo16]*rate_eval.o16__he4_c12
       +Y[jli6]*rate_eval.li6__n_p_he4
       +2*Y[jbe9]*rate_eval.be9__n_he4_he4
       +3*Y[jc12]*rate_eval.c12__he4_he4_he4
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
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__he4_li6
       +rho*Y[jn]*Y[jb10]*rate_eval.n_b10__he4_li7
       +rho*Y[jp]*Y[jb10]*rate_eval.p_b10__he4_be7
       +rho*Y[jn]*Y[jb11]*rate_eval.n_b11__he4_li8
       +rho*Y[jp]*Y[jb12]*rate_eval.p_b12__he4_be9
       +rho*Y[jp]*Y[jc11]*rate_eval.p_c11__he4_b8
       +rho*Y[jn]*Y[jc12]*rate_eval.n_c12__he4_be9
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__he4_b10
       +rho*Y[jp]*Y[jc14]*rate_eval.p_c14__he4_b11
       +rho*Y[jn]*Y[jn13]*rate_eval.n_n13__he4_b10
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__he4_b11
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__he4_c11
       +rho*Y[jn]*Y[jn15]*rate_eval.n_n15__he4_b12
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jn]*Y[jo14]*rate_eval.n_o14__he4_c11
       +rho*Y[jn]*Y[jo15]*rate_eval.n_o15__he4_c12
       +rho*Y[jp]*Y[jo15]*rate_eval.p_o15__he4_n12
       +rho*Y[jn]*Y[jo16]*rate_eval.n_o16__he4_c13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       +5.00000000000000e-01*rho*Y[jt]**2*rate_eval.t_t__n_n_he4
       +rho*Y[jt]*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jp]*Y[jli8]*rate_eval.p_li8__n_he4_he4
       +2*rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       +2*rho*Y[jn]*Y[jb8]*rate_eval.n_b8__p_he4_he4
       +3*rho*Y[jp]*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       +3*rho*Y[jn]*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       +2*rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +2*rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +2*rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +2*rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       )

    dYdt[jli6] = (
       -Y[jli6]*rate_eval.li6__he4_d
       -Y[jli6]*rate_eval.li6__n_p_he4
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__b10
       -rho*Y[jn]*Y[jli6]*rate_eval.n_li6__he4_t
       -rho*Y[jp]*Y[jli6]*rate_eval.p_li6__he4_he3
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       -rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       -rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__p_be9
       +Y[jli7]*rate_eval.li7__n_li6
       +Y[jbe7]*rate_eval.be7__p_li6
       +Y[jb10]*rate_eval.b10__he4_li6
       +rho*Y[jd]*Y[jhe4]*rate_eval.he4_d__li6
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__n_li6
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__p_li6
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__he4_li6
       +rho**2*Y[jn]*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       )

    dYdt[jli7] = (
       -Y[jli7]*rate_eval.li7__n_li6
       -Y[jli7]*rate_eval.li7__he4_t
       -rho*Y[jn]*Y[jli7]*rate_eval.n_li7__li8
       -rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__b11
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__p_li8
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_be9
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__d_li8
       -rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__n_b10
       -rho*Y[jd]*Y[jli7]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       -rho*Y[jhe3]*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       +Y[jli8]*rate_eval.li8__n_li7
       +Y[jb11]*rate_eval.b11__he4_li7
       +rho*Y[jt]*Y[jhe4]*rate_eval.he4_t__li7
       +rho*Y[jn]*Y[jli6]*rate_eval.n_li6__li7
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__p_li7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jp]*Y[jli8]*rate_eval.p_li8__d_li7
       +rho*Y[jd]*Y[jli8]*rate_eval.d_li8__t_li7
       +rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       +rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__t_li7
       +rho*Y[jn]*Y[jb10]*rate_eval.n_b10__he4_li7
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       +2.50000000000000e-01*rho**3*Y[jn]**2*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       )

    dYdt[jli8] = (
       -Y[jli8]*rate_eval.li8__n_li7
       -Y[jli8]*rate_eval.li8__he4_he4__weak__wc12
       -rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__b12
       -rho*Y[jp]*Y[jli8]*rate_eval.p_li8__d_li7
       -rho*Y[jd]*Y[jli8]*rate_eval.d_li8__n_be9
       -rho*Y[jd]*Y[jli8]*rate_eval.d_li8__t_li7
       -rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__n_b11
       -rho*Y[jp]*Y[jli8]*rate_eval.p_li8__n_he4_he4
       +Y[jb12]*rate_eval.b12__he4_li8
       +rho*Y[jn]*Y[jli7]*rate_eval.n_li7__li8
       +rho*Y[jd]*Y[jli7]*rate_eval.d_li7__p_li8
       +rho*Y[jt]*Y[jli7]*rate_eval.t_li7__d_li8
       +rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__d_li8
       +rho*Y[jn]*Y[jb11]*rate_eval.n_b11__he4_li8
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       )

    dYdt[jbe7] = (
       -rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       -Y[jbe7]*rate_eval.be7__p_li6
       -Y[jbe7]*rate_eval.be7__he4_he3
       -rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__c11
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*Y[jbe7]*rate_eval.n_be7__he4_he4
       -rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__p_b10
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       -rho*Y[jt]*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       -rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +Y[jb8]*rate_eval.b8__p_be7
       +Y[jc11]*rate_eval.c11__he4_be7
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       +rho*Y[jp]*Y[jli6]*rate_eval.p_li6__be7
       +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.he4_he4__n_be7
       +rho*Y[jd]*Y[jli6]*rate_eval.d_li6__n_be7
       +rho*Y[jp]*Y[jli7]*rate_eval.p_li7__n_be7
       +rho*Y[jp]*Y[jb10]*rate_eval.p_b10__he4_be7
       +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       +2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       )

    dYdt[jbe9] = (
       -Y[jbe9]*rate_eval.be9__n_he4_he4
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__b10
       -rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__d_li8
       -rho*Y[jn]*Y[jbe9]*rate_eval.n_be9__t_li7
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__he4_li6
       -rho*Y[jt]*Y[jbe9]*rate_eval.t_be9__n_b11
       -rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__n_c12
       -rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__p_b12
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +Y[jb10]*rate_eval.b10__p_be9
       +rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__p_be9
       +rho*Y[jt]*Y[jli7]*rate_eval.t_li7__n_be9
       +rho*Y[jd]*Y[jli8]*rate_eval.d_li8__n_be9
       +rho*Y[jn]*Y[jb11]*rate_eval.n_b11__t_be9
       +rho*Y[jp]*Y[jb12]*rate_eval.p_b12__he4_be9
       +rho*Y[jn]*Y[jc12]*rate_eval.n_c12__he4_be9
       +5.00000000000000e-01*rho**2*Y[jn]*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       +5.00000000000000e-01*rho**2*Y[jd]*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       )

    dYdt[jb8] = (
       -Y[jb8]*rate_eval.b8__p_be7
       -Y[jb8]*rate_eval.b8__he4_he4__weak__wc12
       -rho*Y[jhe4]*Y[jb8]*rate_eval.he4_b8__p_c11
       -rho*Y[jn]*Y[jb8]*rate_eval.n_b8__p_he4_he4
       +rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       +rho*Y[jp]*Y[jc11]*rate_eval.p_c11__he4_b8
       +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       )

    dYdt[jb10] = (
       -Y[jb10]*rate_eval.b10__p_be9
       -Y[jb10]*rate_eval.b10__he4_li6
       -rho*Y[jn]*Y[jb10]*rate_eval.n_b10__b11
       -rho*Y[jp]*Y[jb10]*rate_eval.p_b10__c11
       -rho*Y[jn]*Y[jb10]*rate_eval.n_b10__he4_li7
       -rho*Y[jp]*Y[jb10]*rate_eval.p_b10__he4_be7
       -rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__n_n13
       -rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__p_c13
       +Y[jb11]*rate_eval.b11__n_b10
       +Y[jc11]*rate_eval.c11__p_b10
       +rho*Y[jhe4]*Y[jli6]*rate_eval.he4_li6__b10
       +rho*Y[jp]*Y[jbe9]*rate_eval.p_be9__b10
       +rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__n_b10
       +rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__p_b10
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__he4_b10
       +rho*Y[jn]*Y[jn13]*rate_eval.n_n13__he4_b10
       )

    dYdt[jb11] = (
       -Y[jb11]*rate_eval.b11__n_b10
       -Y[jb11]*rate_eval.b11__he4_li7
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__b12
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__c12
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__t_be9
       -rho*Y[jn]*Y[jb11]*rate_eval.n_b11__he4_li8
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__n_c11
       -rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__n_n14
       -rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__p_c14
       -rho*Y[jp]*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       +Y[jc11]*rate_eval.c11__b11__weak__wc12
       +Y[jb12]*rate_eval.b12__n_b11
       +Y[jc12]*rate_eval.c12__p_b11
       +rho*Y[jhe4]*Y[jli7]*rate_eval.he4_li7__b11
       +rho*Y[jn]*Y[jb10]*rate_eval.n_b10__b11
       +rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__n_b11
       +rho*Y[jt]*Y[jbe9]*rate_eval.t_be9__n_b11
       +rho*Y[jn]*Y[jc11]*rate_eval.n_c11__p_b11
       +rho*Y[jp]*Y[jc14]*rate_eval.p_c14__he4_b11
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__he4_b11
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__p_b11
       )

    dYdt[jb12] = (
       -Y[jb12]*rate_eval.b12__c12__weak__wc17
       -Y[jb12]*rate_eval.b12__n_b11
       -Y[jb12]*rate_eval.b12__he4_li8
       -rho*Y[jp]*Y[jb12]*rate_eval.p_b12__n_c12
       -rho*Y[jp]*Y[jb12]*rate_eval.p_b12__he4_be9
       -rho*Y[jhe4]*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jhe4]*Y[jli8]*rate_eval.he4_li8__b12
       +rho*Y[jn]*Y[jb11]*rate_eval.n_b11__b12
       +rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__p_b12
       +rho*Y[jn]*Y[jc12]*rate_eval.n_c12__p_b12
       +rho*Y[jn]*Y[jn15]*rate_eval.n_n15__he4_b12
       )

    dYdt[jc11] = (
       -Y[jc11]*rate_eval.c11__b11__weak__wc12
       -Y[jc11]*rate_eval.c11__p_b10
       -Y[jc11]*rate_eval.c11__he4_be7
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__c12
       -rho*Y[jp]*Y[jc11]*rate_eval.p_c11__n12
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__p_b11
       -rho*Y[jp]*Y[jc11]*rate_eval.p_c11__he4_b8
       -rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__n_o14
       -rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__p_n14
       -rho*Y[jn]*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       +Y[jc12]*rate_eval.c12__n_c11
       +Y[jn12]*rate_eval.n12__p_c11
       +rho*Y[jhe4]*Y[jbe7]*rate_eval.he4_be7__c11
       +rho*Y[jp]*Y[jb10]*rate_eval.p_b10__c11
       +rho*Y[jhe4]*Y[jb8]*rate_eval.he4_b8__p_c11
       +rho*Y[jp]*Y[jb11]*rate_eval.p_b11__n_c11
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__he4_c11
       +rho*Y[jn]*Y[jo14]*rate_eval.n_o14__he4_c11
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__n_c11
       )

    dYdt[jc12] = (
       -Y[jc12]*rate_eval.c12__n_c11
       -Y[jc12]*rate_eval.c12__p_b11
       -Y[jc12]*rate_eval.c12__he4_he4_he4
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__c13
       -rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__p_b12
       -rho*Y[jn]*Y[jc12]*rate_eval.n_c12__he4_be9
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__n_o15
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       +Y[jb12]*rate_eval.b12__c12__weak__wc17
       +Y[jn12]*rate_eval.n12__c12__weak__wc12
       +Y[jc13]*rate_eval.c13__n_c12
       +Y[jn13]*rate_eval.n13__p_c12
       +Y[jo16]*rate_eval.o16__he4_c12
       +rho*Y[jp]*Y[jb11]*rate_eval.p_b11__c12
       +rho*Y[jn]*Y[jc11]*rate_eval.n_c11__c12
       +rho*Y[jhe4]*Y[jbe9]*rate_eval.he4_be9__n_c12
       +rho*Y[jp]*Y[jb12]*rate_eval.p_b12__n_c12
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jn]*Y[jo15]*rate_eval.n_o15__he4_c12
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       )

    dYdt[jc13] = (
       -Y[jc13]*rate_eval.c13__n_c12
       -rho*Y[jn]*Y[jc13]*rate_eval.n_c13__c14
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n_n13
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__he4_b10
       -rho*Y[jd]*Y[jc13]*rate_eval.d_c13__n_n14
       -rho*Y[jhe4]*Y[jc13]*rate_eval.he4_c13__n_o16
       +Y[jn13]*rate_eval.n13__c13__weak__wc12
       +Y[jc14]*rate_eval.c14__n_c13
       +Y[jn14]*rate_eval.n14__p_c13
       +rho*Y[jn]*Y[jc12]*rate_eval.n_c12__c13
       +rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__p_c13
       +rho*Y[jn]*Y[jn13]*rate_eval.n_n13__p_c13
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__d_c13
       +rho*Y[jn]*Y[jo16]*rate_eval.n_o16__he4_c13
       )

    dYdt[jc14] = (
       -Y[jc14]*rate_eval.c14__n14__weak__wc12
       -Y[jc14]*rate_eval.c14__n_c13
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n15
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n_n14
       -rho*Y[jp]*Y[jc14]*rate_eval.p_c14__he4_b11
       -rho*Y[jd]*Y[jc14]*rate_eval.d_c14__n_n15
       +Y[jn15]*rate_eval.n15__p_c14
       +rho*Y[jn]*Y[jc13]*rate_eval.n_c13__c14
       +rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__p_c14
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__p_c14
       +rho*Y[jn]*Y[jn15]*rate_eval.n_n15__d_c14
       )

    dYdt[jn12] = (
       -Y[jn12]*rate_eval.n12__c12__weak__wc12
       -Y[jn12]*rate_eval.n12__p_c11
       -rho*Y[jhe4]*Y[jn12]*rate_eval.he4_n12__p_o15
       +rho*Y[jp]*Y[jc11]*rate_eval.p_c11__n12
       +rho*Y[jp]*Y[jo15]*rate_eval.p_o15__he4_n12
       )

    dYdt[jn13] = (
       -Y[jn13]*rate_eval.n13__c13__weak__wc12
       -Y[jn13]*rate_eval.n13__p_c12
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__n14
       -rho*Y[jp]*Y[jn13]*rate_eval.p_n13__o14
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__p_c13
       -rho*Y[jn]*Y[jn13]*rate_eval.n_n13__he4_b10
       -rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       +Y[jn14]*rate_eval.n14__n_n13
       +Y[jo14]*rate_eval.o14__p_n13
       +rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jhe4]*Y[jb10]*rate_eval.he4_b10__n_n13
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n_n13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       )

    dYdt[jn14] = (
       -Y[jn14]*rate_eval.n14__n_n13
       -Y[jn14]*rate_eval.n14__p_c13
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__n15
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__p_c14
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__d_c13
       -rho*Y[jn]*Y[jn14]*rate_eval.n_n14__he4_b11
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__n_o14
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__he4_c11
       +Y[jc14]*rate_eval.c14__n14__weak__wc12
       +Y[jo14]*rate_eval.o14__n14__weak__wc12
       +Y[jn15]*rate_eval.n15__n_n14
       +Y[jo15]*rate_eval.o15__p_n14
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       +rho*Y[jn]*Y[jn13]*rate_eval.n_n13__n14
       +rho*Y[jhe4]*Y[jb11]*rate_eval.he4_b11__n_n14
       +rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__p_n14
       +rho*Y[jd]*Y[jc13]*rate_eval.d_c13__n_n14
       +rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n_n14
       +rho*Y[jn]*Y[jo14]*rate_eval.n_o14__p_n14
       )

    dYdt[jn15] = (
       -Y[jn15]*rate_eval.n15__n_n14
       -Y[jn15]*rate_eval.n15__p_c14
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jn]*Y[jn15]*rate_eval.n_n15__d_c14
       -rho*Y[jn]*Y[jn15]*rate_eval.n_n15__he4_b12
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__n_o15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +Y[jo15]*rate_eval.o15__n15__weak__wc12
       +Y[jo16]*rate_eval.o16__p_n15
       +rho*Y[jp]*Y[jc14]*rate_eval.p_c14__n15
       +rho*Y[jn]*Y[jn14]*rate_eval.n_n14__n15
       +rho*Y[jhe4]*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jd]*Y[jc14]*rate_eval.d_c14__n_n15
       +rho*Y[jn]*Y[jo15]*rate_eval.n_o15__p_n15
       )

    dYdt[jo14] = (
       -Y[jo14]*rate_eval.o14__n14__weak__wc12
       -Y[jo14]*rate_eval.o14__p_n13
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__o15
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__p_n14
       -rho*Y[jn]*Y[jo14]*rate_eval.n_o14__he4_c11
       +Y[jo15]*rate_eval.o15__n_o14
       +rho*Y[jp]*Y[jn13]*rate_eval.p_n13__o14
       +rho*Y[jhe4]*Y[jc11]*rate_eval.he4_c11__n_o14
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__n_o14
       )

    dYdt[jo15] = (
       -Y[jo15]*rate_eval.o15__n15__weak__wc12
       -Y[jo15]*rate_eval.o15__n_o14
       -Y[jo15]*rate_eval.o15__p_n14
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__o16
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__p_n15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_o15__he4_c12
       -rho*Y[jp]*Y[jo15]*rate_eval.p_o15__he4_n12
       +Y[jo16]*rate_eval.o16__n_o15
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       +rho*Y[jn]*Y[jo14]*rate_eval.n_o14__o15
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__n_o15
       +rho*Y[jhe4]*Y[jn12]*rate_eval.he4_n12__p_o15
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__n_o15
       )

    dYdt[jo16] = (
       -Y[jo16]*rate_eval.o16__n_o15
       -Y[jo16]*rate_eval.o16__p_n15
       -Y[jo16]*rate_eval.o16__he4_c12
       -rho*Y[jn]*Y[jo16]*rate_eval.n_o16__he4_c13
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       +rho*Y[jn]*Y[jo15]*rate_eval.n_o15__o16
       +rho*Y[jhe4]*Y[jc13]*rate_eval.he4_c13__n_o16
       +rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
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
    be7__li7__weak__electron_capture(rate_eval, tf)
    b12__c12__weak__wc17(rate_eval, tf)
    c11__b11__weak__wc12(rate_eval, tf)
    c14__n14__weak__wc12(rate_eval, tf)
    n12__c12__weak__wc12(rate_eval, tf)
    n13__c13__weak__wc12(rate_eval, tf)
    o14__n14__weak__wc12(rate_eval, tf)
    o15__n15__weak__wc12(rate_eval, tf)
    d__n_p(rate_eval, tf)
    t__n_d(rate_eval, tf)
    he3__p_d(rate_eval, tf)
    he4__n_he3(rate_eval, tf)
    he4__p_t(rate_eval, tf)
    he4__d_d(rate_eval, tf)
    li6__he4_d(rate_eval, tf)
    li7__n_li6(rate_eval, tf)
    li7__he4_t(rate_eval, tf)
    li8__n_li7(rate_eval, tf)
    li8__he4_he4__weak__wc12(rate_eval, tf)
    be7__p_li6(rate_eval, tf)
    be7__he4_he3(rate_eval, tf)
    b8__p_be7(rate_eval, tf)
    b8__he4_he4__weak__wc12(rate_eval, tf)
    b10__p_be9(rate_eval, tf)
    b10__he4_li6(rate_eval, tf)
    b11__n_b10(rate_eval, tf)
    b11__he4_li7(rate_eval, tf)
    b12__n_b11(rate_eval, tf)
    b12__he4_li8(rate_eval, tf)
    c11__p_b10(rate_eval, tf)
    c11__he4_be7(rate_eval, tf)
    c12__n_c11(rate_eval, tf)
    c12__p_b11(rate_eval, tf)
    c13__n_c12(rate_eval, tf)
    c14__n_c13(rate_eval, tf)
    n12__p_c11(rate_eval, tf)
    n13__p_c12(rate_eval, tf)
    n14__n_n13(rate_eval, tf)
    n14__p_c13(rate_eval, tf)
    n15__n_n14(rate_eval, tf)
    n15__p_c14(rate_eval, tf)
    o14__p_n13(rate_eval, tf)
    o15__n_o14(rate_eval, tf)
    o15__p_n14(rate_eval, tf)
    o16__n_o15(rate_eval, tf)
    o16__p_n15(rate_eval, tf)
    o16__he4_c12(rate_eval, tf)
    li6__n_p_he4(rate_eval, tf)
    be9__n_he4_he4(rate_eval, tf)
    c12__he4_he4_he4(rate_eval, tf)
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
    he4_li6__b10(rate_eval, tf)
    n_li7__li8(rate_eval, tf)
    he4_li7__b11(rate_eval, tf)
    he4_li8__b12(rate_eval, tf)
    p_be7__b8(rate_eval, tf)
    he4_be7__c11(rate_eval, tf)
    p_be9__b10(rate_eval, tf)
    n_b10__b11(rate_eval, tf)
    p_b10__c11(rate_eval, tf)
    n_b11__b12(rate_eval, tf)
    p_b11__c12(rate_eval, tf)
    n_c11__c12(rate_eval, tf)
    p_c11__n12(rate_eval, tf)
    n_c12__c13(rate_eval, tf)
    p_c12__n13(rate_eval, tf)
    he4_c12__o16(rate_eval, tf)
    n_c13__c14(rate_eval, tf)
    p_c13__n14(rate_eval, tf)
    p_c14__n15(rate_eval, tf)
    n_n13__n14(rate_eval, tf)
    p_n13__o14(rate_eval, tf)
    n_n14__n15(rate_eval, tf)
    p_n14__o15(rate_eval, tf)
    p_n15__o16(rate_eval, tf)
    n_o14__o15(rate_eval, tf)
    n_o15__o16(rate_eval, tf)
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
    he4_li6__p_be9(rate_eval, tf)
    p_li7__n_be7(rate_eval, tf)
    p_li7__d_li6(rate_eval, tf)
    p_li7__he4_he4(rate_eval, tf)
    d_li7__p_li8(rate_eval, tf)
    t_li7__n_be9(rate_eval, tf)
    t_li7__d_li8(rate_eval, tf)
    he4_li7__n_b10(rate_eval, tf)
    p_li8__d_li7(rate_eval, tf)
    d_li8__n_be9(rate_eval, tf)
    d_li8__t_li7(rate_eval, tf)
    he4_li8__n_b11(rate_eval, tf)
    n_be7__p_li7(rate_eval, tf)
    n_be7__d_li6(rate_eval, tf)
    n_be7__he4_he4(rate_eval, tf)
    he4_be7__p_b10(rate_eval, tf)
    n_be9__d_li8(rate_eval, tf)
    n_be9__t_li7(rate_eval, tf)
    p_be9__he4_li6(rate_eval, tf)
    t_be9__n_b11(rate_eval, tf)
    he4_be9__n_c12(rate_eval, tf)
    he4_be9__p_b12(rate_eval, tf)
    he4_b8__p_c11(rate_eval, tf)
    n_b10__he4_li7(rate_eval, tf)
    p_b10__he4_be7(rate_eval, tf)
    he4_b10__n_n13(rate_eval, tf)
    he4_b10__p_c13(rate_eval, tf)
    n_b11__t_be9(rate_eval, tf)
    n_b11__he4_li8(rate_eval, tf)
    p_b11__n_c11(rate_eval, tf)
    he4_b11__n_n14(rate_eval, tf)
    he4_b11__p_c14(rate_eval, tf)
    p_b12__n_c12(rate_eval, tf)
    p_b12__he4_be9(rate_eval, tf)
    he4_b12__n_n15(rate_eval, tf)
    n_c11__p_b11(rate_eval, tf)
    p_c11__he4_b8(rate_eval, tf)
    he4_c11__n_o14(rate_eval, tf)
    he4_c11__p_n14(rate_eval, tf)
    n_c12__p_b12(rate_eval, tf)
    n_c12__he4_be9(rate_eval, tf)
    he4_c12__n_o15(rate_eval, tf)
    he4_c12__p_n15(rate_eval, tf)
    p_c13__n_n13(rate_eval, tf)
    p_c13__he4_b10(rate_eval, tf)
    d_c13__n_n14(rate_eval, tf)
    he4_c13__n_o16(rate_eval, tf)
    p_c14__n_n14(rate_eval, tf)
    p_c14__he4_b11(rate_eval, tf)
    d_c14__n_n15(rate_eval, tf)
    he4_n12__p_o15(rate_eval, tf)
    n_n13__p_c13(rate_eval, tf)
    n_n13__he4_b10(rate_eval, tf)
    he4_n13__p_o16(rate_eval, tf)
    n_n14__p_c14(rate_eval, tf)
    n_n14__d_c13(rate_eval, tf)
    n_n14__he4_b11(rate_eval, tf)
    p_n14__n_o14(rate_eval, tf)
    p_n14__he4_c11(rate_eval, tf)
    n_n15__d_c14(rate_eval, tf)
    n_n15__he4_b12(rate_eval, tf)
    p_n15__n_o15(rate_eval, tf)
    p_n15__he4_c12(rate_eval, tf)
    n_o14__p_n14(rate_eval, tf)
    n_o14__he4_c11(rate_eval, tf)
    n_o15__p_n15(rate_eval, tf)
    n_o15__he4_c12(rate_eval, tf)
    p_o15__he4_n12(rate_eval, tf)
    n_o16__he4_c13(rate_eval, tf)
    p_o16__he4_n13(rate_eval, tf)
    p_d__n_p_p(rate_eval, tf)
    t_t__n_n_he4(rate_eval, tf)
    t_he3__n_p_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    d_li7__n_he4_he4(rate_eval, tf)
    p_li8__n_he4_he4(rate_eval, tf)
    d_be7__p_he4_he4(rate_eval, tf)
    p_be9__d_he4_he4(rate_eval, tf)
    n_b8__p_he4_he4(rate_eval, tf)
    p_b11__he4_he4_he4(rate_eval, tf)
    n_c11__he4_he4_he4(rate_eval, tf)
    t_li7__n_n_he4_he4(rate_eval, tf)
    he3_li7__n_p_he4_he4(rate_eval, tf)
    t_be7__n_p_he4_he4(rate_eval, tf)
    he3_be7__p_p_he4_he4(rate_eval, tf)
    p_be9__n_p_he4_he4(rate_eval, tf)
    n_p_he4__li6(rate_eval, tf)
    n_he4_he4__be9(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)
    n_p_p__p_d(rate_eval, tf)
    n_n_he4__t_t(rate_eval, tf)
    n_p_he4__t_he3(rate_eval, tf)
    p_p_he4__he3_he3(rate_eval, tf)
    n_he4_he4__p_li8(rate_eval, tf)
    n_he4_he4__d_li7(rate_eval, tf)
    p_he4_he4__n_b8(rate_eval, tf)
    p_he4_he4__d_be7(rate_eval, tf)
    d_he4_he4__p_be9(rate_eval, tf)
    he4_he4_he4__n_c11(rate_eval, tf)
    he4_he4_he4__p_b11(rate_eval, tf)
    n_n_he4_he4__t_li7(rate_eval, tf)
    n_p_he4_he4__he3_li7(rate_eval, tf)
    n_p_he4_he4__t_be7(rate_eval, tf)
    n_p_he4_he4__p_be9(rate_eval, tf)
    p_p_he4_he4__he3_be7(rate_eval, tf)

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

        scn_fac = ScreenFactors(2, 4, 3, 6)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li6__b10 *= scor
        rate_eval.he4_li6__p_be9 *= scor

        scn_fac = ScreenFactors(2, 4, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li7__b11 *= scor
        rate_eval.he4_li7__n_b10 *= scor

        scn_fac = ScreenFactors(2, 4, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_li8__b12 *= scor
        rate_eval.he4_li8__n_b11 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be7__b8 *= scor

        scn_fac = ScreenFactors(2, 4, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_be7__c11 *= scor
        rate_eval.he4_be7__p_b10 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be9__b10 *= scor
        rate_eval.p_be9__he4_li6 *= scor
        rate_eval.p_be9__d_he4_he4 *= scor
        rate_eval.p_be9__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 10)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b10__c11 *= scor
        rate_eval.p_b10__he4_be7 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b11__c12 *= scor
        rate_eval.p_b11__n_c11 *= scor
        rate_eval.p_b11__he4_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c11__n12 *= scor
        rate_eval.p_c11__he4_b8 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c12__n13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c12__o16 *= scor
        rate_eval.he4_c12__n_o15 *= scor
        rate_eval.he4_c12__p_n15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c13__n14 *= scor
        rate_eval.p_c13__n_n13 *= scor
        rate_eval.p_c13__he4_b10 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c14__n15 *= scor
        rate_eval.p_c14__n_n14 *= scor
        rate_eval.p_c14__he4_b11 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n13__o14 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n14__o15 *= scor
        rate_eval.p_n14__n_o14 *= scor
        rate_eval.p_n14__he4_c11 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n15__o16 *= scor
        rate_eval.p_n15__n_o15 *= scor
        rate_eval.p_n15__he4_c12 *= scor

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
        rate_eval.n_he4_he4__be9 *= scor
        rate_eval.n_he4_he4__p_li8 *= scor
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

        scn_fac = ScreenFactors(1, 2, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li7__p_li8 *= scor
        rate_eval.d_li7__n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_li7__n_be9 *= scor
        rate_eval.t_li7__d_li8 *= scor
        rate_eval.t_li7__n_n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li8__d_li7 *= scor
        rate_eval.p_li8__n_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 3, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_li8__n_be9 *= scor
        rate_eval.d_li8__t_li7 *= scor

        scn_fac = ScreenFactors(1, 3, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_be9__n_b11 *= scor

        scn_fac = ScreenFactors(2, 4, 4, 9)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_be9__n_c12 *= scor
        rate_eval.he4_be9__p_b12 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 8)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b8__p_c11 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 10)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b10__n_n13 *= scor
        rate_eval.he4_b10__p_c13 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b11__n_n14 *= scor
        rate_eval.he4_b11__p_c14 *= scor

        scn_fac = ScreenFactors(1, 1, 5, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_b12__n_c12 *= scor
        rate_eval.p_b12__he4_be9 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_b12__n_n15 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c11__n_o14 *= scor
        rate_eval.he4_c11__p_n14 *= scor

        scn_fac = ScreenFactors(1, 2, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_c13__n_n14 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c13__n_o16 *= scor

        scn_fac = ScreenFactors(1, 2, 6, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_c14__n_n15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n12__p_o15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n13__p_o16 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o15__he4_n12 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o16__he4_n13 *= scor

        scn_fac = ScreenFactors(1, 3, 1, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_t__n_n_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_be7__p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_li7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(1, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.t_be7__n_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_be7__p_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2
        rate_eval.he4_he4_he4__n_c11 *= scor * scor2
        rate_eval.he4_he4_he4__p_b11 *= scor * scor2

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_he4__he3_he3 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he4_he4__n_b8 *= scor
        rate_eval.p_he4_he4__d_be7 *= scor
        rate_eval.n_p_he4_he4__he3_li7 *= scor
        rate_eval.n_p_he4_he4__t_be7 *= scor
        rate_eval.n_p_he4_he4__p_be9 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he4_he4__p_be9 *= scor

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -rho*Y[jd]*rate_eval.n_d__t
       -rho*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jli7]*rate_eval.n_li7__li8
       -rho*Y[jb10]*rate_eval.n_b10__b11
       -rho*Y[jb11]*rate_eval.n_b11__b12
       -rho*Y[jc11]*rate_eval.n_c11__c12
       -rho*Y[jc12]*rate_eval.n_c12__c13
       -rho*Y[jc13]*rate_eval.n_c13__c14
       -rho*Y[jn13]*rate_eval.n_n13__n14
       -rho*Y[jn14]*rate_eval.n_n14__n15
       -rho*Y[jo14]*rate_eval.n_o14__o15
       -rho*Y[jo15]*rate_eval.n_o15__o16
       -rho*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jhe3]*rate_eval.n_he3__d_d
       -rho*Y[jhe4]*rate_eval.n_he4__d_t
       -rho*Y[jli6]*rate_eval.n_li6__he4_t
       -rho*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jbe7]*rate_eval.n_be7__he4_he4
       -rho*Y[jbe9]*rate_eval.n_be9__d_li8
       -rho*Y[jbe9]*rate_eval.n_be9__t_li7
       -rho*Y[jb10]*rate_eval.n_b10__he4_li7
       -rho*Y[jb11]*rate_eval.n_b11__t_be9
       -rho*Y[jb11]*rate_eval.n_b11__he4_li8
       -rho*Y[jc11]*rate_eval.n_c11__p_b11
       -rho*Y[jc12]*rate_eval.n_c12__p_b12
       -rho*Y[jc12]*rate_eval.n_c12__he4_be9
       -rho*Y[jn13]*rate_eval.n_n13__p_c13
       -rho*Y[jn13]*rate_eval.n_n13__he4_b10
       -rho*Y[jn14]*rate_eval.n_n14__p_c14
       -rho*Y[jn14]*rate_eval.n_n14__d_c13
       -rho*Y[jn14]*rate_eval.n_n14__he4_b11
       -rho*Y[jn15]*rate_eval.n_n15__d_c14
       -rho*Y[jn15]*rate_eval.n_n15__he4_b12
       -rho*Y[jo14]*rate_eval.n_o14__p_n14
       -rho*Y[jo14]*rate_eval.n_o14__he4_c11
       -rho*Y[jo15]*rate_eval.n_o15__p_n15
       -rho*Y[jo15]*rate_eval.n_o15__he4_c12
       -rho*Y[jo16]*rate_eval.n_o16__he4_c13
       -rho*Y[jb8]*rate_eval.n_b8__p_he4_he4
       -rho*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       -2*5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*2.50000000000000e-01*rho**3*2*Y[jn]*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -rate_eval.n__p
       )

    jac[jn, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       +rho*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jli7]*rate_eval.p_li7__n_be7
       +rho*Y[jb11]*rate_eval.p_b11__n_c11
       +rho*Y[jb12]*rate_eval.p_b12__n_c12
       +rho*Y[jc13]*rate_eval.p_c13__n_n13
       +rho*Y[jc14]*rate_eval.p_c14__n_n14
       +rho*Y[jn14]*rate_eval.p_n14__n_o14
       +rho*Y[jn15]*rate_eval.p_n15__n_o15
       +rho*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jli8]*rate_eval.p_li8__n_he4_he4
       +rho*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       +rate_eval.p__n
       )

    jac[jn, jd] = (
       -rho*Y[jn]*rate_eval.n_d__t
       +rate_eval.d__n_p
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__n_he3
       +rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jli6]*rate_eval.d_li6__n_be7
       +rho*Y[jli8]*rate_eval.d_li8__n_be9
       +rho*Y[jc13]*rate_eval.d_c13__n_n14
       +rho*Y[jc14]*rate_eval.d_c14__n_n15
       +rho*Y[jp]*rate_eval.p_d__n_p_p
       +rho*Y[jli7]*rate_eval.d_li7__n_he4_he4
       )

    jac[jn, jt] = (
       +rate_eval.t__n_d
       +rho*Y[jp]*rate_eval.p_t__n_he3
       +rho*Y[jd]*rate_eval.d_t__n_he4
       +rho*Y[jhe4]*rate_eval.he4_t__n_li6
       +rho*Y[jli7]*rate_eval.t_li7__n_be9
       +rho*Y[jbe9]*rate_eval.t_be9__n_b11
       +2*5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +2*rho*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +rho*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jn, jhe3] = (
       -rho*Y[jn]*rate_eval.n_he3__he4
       -rho*Y[jn]*rate_eval.n_he3__p_t
       -rho*Y[jn]*rate_eval.n_he3__d_d
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +rho*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jn, jhe4] = (
       -rho*Y[jn]*rate_eval.n_he4__d_t
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__li6
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__be9
       -2*5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__p_li8
       -5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__d_li7
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*2*Y[jhe4]*rate_eval.n_n_he4_he4__t_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__p_be9
       +rate_eval.he4__n_he3
       +rho*Y[jt]*rate_eval.he4_t__n_li6
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__n_be7
       +rho*Y[jli7]*rate_eval.he4_li7__n_b10
       +rho*Y[jli8]*rate_eval.he4_li8__n_b11
       +rho*Y[jbe9]*rate_eval.he4_be9__n_c12
       +rho*Y[jb10]*rate_eval.he4_b10__n_n13
       +rho*Y[jb11]*rate_eval.he4_b11__n_n14
       +rho*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jc11]*rate_eval.he4_c11__n_o14
       +rho*Y[jc12]*rate_eval.he4_c12__n_o15
       +rho*Y[jc13]*rate_eval.he4_c13__n_o16
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__n_b8
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__n_c11
       )

    jac[jn, jli6] = (
       -rho*Y[jn]*rate_eval.n_li6__li7
       -rho*Y[jn]*rate_eval.n_li6__he4_t
       +rate_eval.li6__n_p_he4
       +rho*Y[jd]*rate_eval.d_li6__n_be7
       )

    jac[jn, jli7] = (
       -rho*Y[jn]*rate_eval.n_li7__li8
       +rate_eval.li7__n_li6
       +rho*Y[jp]*rate_eval.p_li7__n_be7
       +rho*Y[jt]*rate_eval.t_li7__n_be9
       +rho*Y[jhe4]*rate_eval.he4_li7__n_b10
       +rho*Y[jd]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jt]*rate_eval.t_li7__n_n_he4_he4
       +rho*Y[jhe3]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jn, jli8] = (
       +rate_eval.li8__n_li7
       +rho*Y[jd]*rate_eval.d_li8__n_be9
       +rho*Y[jhe4]*rate_eval.he4_li8__n_b11
       +rho*Y[jp]*rate_eval.p_li8__n_he4_he4
       )

    jac[jn, jbe7] = (
       -rho*Y[jn]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*rate_eval.n_be7__he4_he4
       +rho*Y[jt]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jn, jbe9] = (
       -rho*Y[jn]*rate_eval.n_be9__d_li8
       -rho*Y[jn]*rate_eval.n_be9__t_li7
       +rate_eval.be9__n_he4_he4
       +rho*Y[jt]*rate_eval.t_be9__n_b11
       +rho*Y[jhe4]*rate_eval.he4_be9__n_c12
       +rho*Y[jp]*rate_eval.p_be9__n_p_he4_he4
       )

    jac[jn, jb8] = (
       -rho*Y[jn]*rate_eval.n_b8__p_he4_he4
       )

    jac[jn, jb10] = (
       -rho*Y[jn]*rate_eval.n_b10__b11
       -rho*Y[jn]*rate_eval.n_b10__he4_li7
       +rho*Y[jhe4]*rate_eval.he4_b10__n_n13
       )

    jac[jn, jb11] = (
       -rho*Y[jn]*rate_eval.n_b11__b12
       -rho*Y[jn]*rate_eval.n_b11__t_be9
       -rho*Y[jn]*rate_eval.n_b11__he4_li8
       +rate_eval.b11__n_b10
       +rho*Y[jp]*rate_eval.p_b11__n_c11
       +rho*Y[jhe4]*rate_eval.he4_b11__n_n14
       )

    jac[jn, jb12] = (
       +rate_eval.b12__n_b11
       +rho*Y[jp]*rate_eval.p_b12__n_c12
       +rho*Y[jhe4]*rate_eval.he4_b12__n_n15
       )

    jac[jn, jc11] = (
       -rho*Y[jn]*rate_eval.n_c11__c12
       -rho*Y[jn]*rate_eval.n_c11__p_b11
       -rho*Y[jn]*rate_eval.n_c11__he4_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_c11__n_o14
       )

    jac[jn, jc12] = (
       -rho*Y[jn]*rate_eval.n_c12__c13
       -rho*Y[jn]*rate_eval.n_c12__p_b12
       -rho*Y[jn]*rate_eval.n_c12__he4_be9
       +rate_eval.c12__n_c11
       +rho*Y[jhe4]*rate_eval.he4_c12__n_o15
       )

    jac[jn, jc13] = (
       -rho*Y[jn]*rate_eval.n_c13__c14
       +rate_eval.c13__n_c12
       +rho*Y[jp]*rate_eval.p_c13__n_n13
       +rho*Y[jd]*rate_eval.d_c13__n_n14
       +rho*Y[jhe4]*rate_eval.he4_c13__n_o16
       )

    jac[jn, jc14] = (
       +rate_eval.c14__n_c13
       +rho*Y[jp]*rate_eval.p_c14__n_n14
       +rho*Y[jd]*rate_eval.d_c14__n_n15
       )

    jac[jn, jn13] = (
       -rho*Y[jn]*rate_eval.n_n13__n14
       -rho*Y[jn]*rate_eval.n_n13__p_c13
       -rho*Y[jn]*rate_eval.n_n13__he4_b10
       )

    jac[jn, jn14] = (
       -rho*Y[jn]*rate_eval.n_n14__n15
       -rho*Y[jn]*rate_eval.n_n14__p_c14
       -rho*Y[jn]*rate_eval.n_n14__d_c13
       -rho*Y[jn]*rate_eval.n_n14__he4_b11
       +rate_eval.n14__n_n13
       +rho*Y[jp]*rate_eval.p_n14__n_o14
       )

    jac[jn, jn15] = (
       -rho*Y[jn]*rate_eval.n_n15__d_c14
       -rho*Y[jn]*rate_eval.n_n15__he4_b12
       +rate_eval.n15__n_n14
       +rho*Y[jp]*rate_eval.p_n15__n_o15
       )

    jac[jn, jo14] = (
       -rho*Y[jn]*rate_eval.n_o14__o15
       -rho*Y[jn]*rate_eval.n_o14__p_n14
       -rho*Y[jn]*rate_eval.n_o14__he4_c11
       )

    jac[jn, jo15] = (
       -rho*Y[jn]*rate_eval.n_o15__o16
       -rho*Y[jn]*rate_eval.n_o15__p_n15
       -rho*Y[jn]*rate_eval.n_o15__he4_c12
       +rate_eval.o15__n_o14
       )

    jac[jn, jo16] = (
       -rho*Y[jn]*rate_eval.n_o16__he4_c13
       +rate_eval.o16__n_o15
       )

    jac[jp, jn] = (
       -rho*Y[jp]*rate_eval.n_p__d
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       +rho*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jbe7]*rate_eval.n_be7__p_li7
       +rho*Y[jc11]*rate_eval.n_c11__p_b11
       +rho*Y[jc12]*rate_eval.n_c12__p_b12
       +rho*Y[jn13]*rate_eval.n_n13__p_c13
       +rho*Y[jn14]*rate_eval.n_n14__p_c14
       +rho*Y[jo14]*rate_eval.n_o14__p_n14
       +rho*Y[jo15]*rate_eval.n_o15__p_n15
       +rho*Y[jb8]*rate_eval.n_b8__p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       +rate_eval.n__p
       )

    jac[jp, jp] = (
       -rho*Y[jn]*rate_eval.n_p__d
       -2*5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jbe9]*rate_eval.p_be9__b10
       -rho*Y[jb10]*rate_eval.p_b10__c11
       -rho*Y[jb11]*rate_eval.p_b11__c12
       -rho*Y[jc11]*rate_eval.p_c11__n12
       -rho*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jc14]*rate_eval.p_c14__n15
       -rho*Y[jn13]*rate_eval.p_n13__o14
       -rho*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jt]*rate_eval.p_t__d_d
       -rho*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho*Y[jli6]*rate_eval.p_li6__he4_he3
       -rho*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jli7]*rate_eval.p_li7__he4_he4
       -rho*Y[jli8]*rate_eval.p_li8__d_li7
       -rho*Y[jbe9]*rate_eval.p_be9__he4_li6
       -rho*Y[jb10]*rate_eval.p_b10__he4_be7
       -rho*Y[jb11]*rate_eval.p_b11__n_c11
       -rho*Y[jb12]*rate_eval.p_b12__n_c12
       -rho*Y[jb12]*rate_eval.p_b12__he4_be9
       -rho*Y[jc11]*rate_eval.p_c11__he4_b8
       -rho*Y[jc13]*rate_eval.p_c13__n_n13
       -rho*Y[jc13]*rate_eval.p_c13__he4_b10
       -rho*Y[jc14]*rate_eval.p_c14__n_n14
       -rho*Y[jc14]*rate_eval.p_c14__he4_b11
       -rho*Y[jn14]*rate_eval.p_n14__n_o14
       -rho*Y[jn14]*rate_eval.p_n14__he4_c11
       -rho*Y[jn15]*rate_eval.p_n15__n_o15
       -rho*Y[jn15]*rate_eval.p_n15__he4_c12
       -rho*Y[jo15]*rate_eval.p_o15__he4_n12
       -rho*Y[jo16]*rate_eval.p_o16__he4_n13
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       -rho*Y[jli8]*rate_eval.p_li8__n_he4_he4
       -rho*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       -rho*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       -rate_eval.p__n
       +2*rho*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       )

    jac[jp, jd] = (
       -rho*Y[jp]*rate_eval.p_d__he3
       -rho*Y[jp]*rate_eval.p_d__n_p_p
       +rate_eval.d__n_p
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jli7]*rate_eval.d_li7__p_li8
       +2*rho*Y[jp]*rate_eval.p_d__n_p_p
       +rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       )

    jac[jp, jt] = (
       -rho*Y[jp]*rate_eval.p_t__he4
       -rho*Y[jp]*rate_eval.p_t__n_he3
       -rho*Y[jp]*rate_eval.p_t__d_d
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +rho*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jp, jhe3] = (
       -rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rate_eval.he3__p_d
       +rho*Y[jn]*rate_eval.n_he3__p_t
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +rho*Y[jhe4]*rate_eval.he4_he3__p_li6
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       +rho*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +2*rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jp, jhe4] = (
       -rho*Y[jp]*rate_eval.p_he4__d_he3
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__li6
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
       -5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__n_b8
       -5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__d_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__he3_li7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__t_be7
       -5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_he4_he4__he3_be7
       +rate_eval.he4__p_t
       +rho*Y[jhe3]*rate_eval.he4_he3__p_li6
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__p_li7
       +rho*Y[jli6]*rate_eval.he4_li6__p_be9
       +rho*Y[jbe7]*rate_eval.he4_be7__p_b10
       +rho*Y[jbe9]*rate_eval.he4_be9__p_b12
       +rho*Y[jb8]*rate_eval.he4_b8__p_c11
       +rho*Y[jb10]*rate_eval.he4_b10__p_c13
       +rho*Y[jb11]*rate_eval.he4_b11__p_c14
       +rho*Y[jc11]*rate_eval.he4_c11__p_n14
       +rho*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jn12]*rate_eval.he4_n12__p_o15
       +rho*Y[jn13]*rate_eval.he4_n13__p_o16
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__p_li8
       +5.00000000000000e-01*rho**2*Y[jd]*2*Y[jhe4]*rate_eval.d_he4_he4__p_be9
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__p_b11
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__p_be9
       )

    jac[jp, jli6] = (
       -rho*Y[jp]*rate_eval.p_li6__be7
       -rho*Y[jp]*rate_eval.p_li6__he4_he3
       +rate_eval.li6__n_p_he4
       +rho*Y[jd]*rate_eval.d_li6__p_li7
       +rho*Y[jhe4]*rate_eval.he4_li6__p_be9
       )

    jac[jp, jli7] = (
       -rho*Y[jp]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*rate_eval.p_li7__he4_he4
       +rho*Y[jd]*rate_eval.d_li7__p_li8
       +rho*Y[jhe3]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jp, jli8] = (
       -rho*Y[jp]*rate_eval.p_li8__d_li7
       -rho*Y[jp]*rate_eval.p_li8__n_he4_he4
       )

    jac[jp, jbe7] = (
       -rho*Y[jp]*rate_eval.p_be7__b8
       +rate_eval.be7__p_li6
       +rho*Y[jn]*rate_eval.n_be7__p_li7
       +rho*Y[jhe4]*rate_eval.he4_be7__p_b10
       +rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       +rho*Y[jt]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jp, jbe9] = (
       -rho*Y[jp]*rate_eval.p_be9__b10
       -rho*Y[jp]*rate_eval.p_be9__he4_li6
       -rho*Y[jp]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jp]*rate_eval.p_be9__n_p_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_be9__p_b12
       +rho*Y[jp]*rate_eval.p_be9__n_p_he4_he4
       )

    jac[jp, jb8] = (
       +rate_eval.b8__p_be7
       +rho*Y[jhe4]*rate_eval.he4_b8__p_c11
       +rho*Y[jn]*rate_eval.n_b8__p_he4_he4
       )

    jac[jp, jb10] = (
       -rho*Y[jp]*rate_eval.p_b10__c11
       -rho*Y[jp]*rate_eval.p_b10__he4_be7
       +rate_eval.b10__p_be9
       +rho*Y[jhe4]*rate_eval.he4_b10__p_c13
       )

    jac[jp, jb11] = (
       -rho*Y[jp]*rate_eval.p_b11__c12
       -rho*Y[jp]*rate_eval.p_b11__n_c11
       -rho*Y[jp]*rate_eval.p_b11__he4_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_b11__p_c14
       )

    jac[jp, jb12] = (
       -rho*Y[jp]*rate_eval.p_b12__n_c12
       -rho*Y[jp]*rate_eval.p_b12__he4_be9
       )

    jac[jp, jc11] = (
       -rho*Y[jp]*rate_eval.p_c11__n12
       -rho*Y[jp]*rate_eval.p_c11__he4_b8
       +rate_eval.c11__p_b10
       +rho*Y[jn]*rate_eval.n_c11__p_b11
       +rho*Y[jhe4]*rate_eval.he4_c11__p_n14
       )

    jac[jp, jc12] = (
       -rho*Y[jp]*rate_eval.p_c12__n13
       +rate_eval.c12__p_b11
       +rho*Y[jn]*rate_eval.n_c12__p_b12
       +rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       )

    jac[jp, jc13] = (
       -rho*Y[jp]*rate_eval.p_c13__n14
       -rho*Y[jp]*rate_eval.p_c13__n_n13
       -rho*Y[jp]*rate_eval.p_c13__he4_b10
       )

    jac[jp, jc14] = (
       -rho*Y[jp]*rate_eval.p_c14__n15
       -rho*Y[jp]*rate_eval.p_c14__n_n14
       -rho*Y[jp]*rate_eval.p_c14__he4_b11
       )

    jac[jp, jn12] = (
       +rate_eval.n12__p_c11
       +rho*Y[jhe4]*rate_eval.he4_n12__p_o15
       )

    jac[jp, jn13] = (
       -rho*Y[jp]*rate_eval.p_n13__o14
       +rate_eval.n13__p_c12
       +rho*Y[jn]*rate_eval.n_n13__p_c13
       +rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jp, jn14] = (
       -rho*Y[jp]*rate_eval.p_n14__o15
       -rho*Y[jp]*rate_eval.p_n14__n_o14
       -rho*Y[jp]*rate_eval.p_n14__he4_c11
       +rate_eval.n14__p_c13
       +rho*Y[jn]*rate_eval.n_n14__p_c14
       )

    jac[jp, jn15] = (
       -rho*Y[jp]*rate_eval.p_n15__o16
       -rho*Y[jp]*rate_eval.p_n15__n_o15
       -rho*Y[jp]*rate_eval.p_n15__he4_c12
       +rate_eval.n15__p_c14
       )

    jac[jp, jo14] = (
       +rate_eval.o14__p_n13
       +rho*Y[jn]*rate_eval.n_o14__p_n14
       )

    jac[jp, jo15] = (
       -rho*Y[jp]*rate_eval.p_o15__he4_n12
       +rate_eval.o15__p_n14
       +rho*Y[jn]*rate_eval.n_o15__p_n15
       )

    jac[jp, jo16] = (
       -rho*Y[jp]*rate_eval.p_o16__he4_n13
       +rate_eval.o16__p_n15
       )

    jac[jd, jn] = (
       -rho*Y[jd]*rate_eval.n_d__t
       +rho*Y[jp]*rate_eval.n_p__d
       +2*rho*Y[jhe3]*rate_eval.n_he3__d_d
       +rho*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jbe7]*rate_eval.n_be7__d_li6
       +rho*Y[jbe9]*rate_eval.n_be9__d_li8
       +rho*Y[jn14]*rate_eval.n_n14__d_c13
       +rho*Y[jn15]*rate_eval.n_n15__d_c14
       +5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       )

    jac[jd, jp] = (
       -rho*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jd]*rate_eval.p_d__n_p_p
       +rho*Y[jn]*rate_eval.n_p__d
       +5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       +2*rho*Y[jt]*rate_eval.p_t__d_d
       +rho*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho*Y[jli7]*rate_eval.p_li7__d_li6
       +rho*Y[jli8]*rate_eval.p_li8__d_li7
       +rho*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jp]*rate_eval.n_p_p__p_d
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       )

    jac[jd, jd] = (
       -rate_eval.d__n_p
       -rho*Y[jn]*rate_eval.n_d__t
       -rho*Y[jp]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       -rho*Y[jhe4]*rate_eval.he4_d__li6
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__n_he3
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       -rho*Y[jt]*rate_eval.d_t__n_he4
       -rho*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jhe4]*rate_eval.d_he4__t_he3
       -rho*Y[jli6]*rate_eval.d_li6__n_be7
       -rho*Y[jli6]*rate_eval.d_li6__p_li7
       -rho*Y[jli7]*rate_eval.d_li7__p_li8
       -rho*Y[jli8]*rate_eval.d_li8__n_be9
       -rho*Y[jli8]*rate_eval.d_li8__t_li7
       -rho*Y[jc13]*rate_eval.d_c13__n_n14
       -rho*Y[jc14]*rate_eval.d_c14__n_n15
       -rho*Y[jp]*rate_eval.p_d__n_p_p
       -rho*Y[jli7]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       )

    jac[jd, jt] = (
       -rho*Y[jd]*rate_eval.d_t__n_he4
       +rate_eval.t__n_d
       +2*rho*Y[jp]*rate_eval.p_t__d_d
       +rho*Y[jhe3]*rate_eval.t_he3__d_he4
       +rho*Y[jli7]*rate_eval.t_li7__d_li8
       )

    jac[jd, jhe3] = (
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       +rate_eval.he3__p_d
       +2*rho*Y[jn]*rate_eval.n_he3__d_d
       +rho*Y[jt]*rate_eval.t_he3__d_he4
       )

    jac[jd, jhe4] = (
       -rho*Y[jd]*rate_eval.he4_d__li6
       -rho*Y[jd]*rate_eval.d_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jd]*2*Y[jhe4]*rate_eval.d_he4_he4__p_be9
       +2*rate_eval.he4__d_d
       +rho*Y[jn]*rate_eval.n_he4__d_t
       +rho*Y[jp]*rate_eval.p_he4__d_he3
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__d_li7
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__d_be7
       )

    jac[jd, jli6] = (
       -rho*Y[jd]*rate_eval.d_li6__n_be7
       -rho*Y[jd]*rate_eval.d_li6__p_li7
       +rate_eval.li6__he4_d
       )

    jac[jd, jli7] = (
       -rho*Y[jd]*rate_eval.d_li7__p_li8
       -rho*Y[jd]*rate_eval.d_li7__n_he4_he4
       +rho*Y[jp]*rate_eval.p_li7__d_li6
       +rho*Y[jt]*rate_eval.t_li7__d_li8
       )

    jac[jd, jli8] = (
       -rho*Y[jd]*rate_eval.d_li8__n_be9
       -rho*Y[jd]*rate_eval.d_li8__t_li7
       +rho*Y[jp]*rate_eval.p_li8__d_li7
       )

    jac[jd, jbe7] = (
       -rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       +rho*Y[jn]*rate_eval.n_be7__d_li6
       )

    jac[jd, jbe9] = (
       +rho*Y[jn]*rate_eval.n_be9__d_li8
       +rho*Y[jp]*rate_eval.p_be9__d_he4_he4
       )

    jac[jd, jc13] = (
       -rho*Y[jd]*rate_eval.d_c13__n_n14
       )

    jac[jd, jc14] = (
       -rho*Y[jd]*rate_eval.d_c14__n_n15
       )

    jac[jd, jn14] = (
       +rho*Y[jn]*rate_eval.n_n14__d_c13
       )

    jac[jd, jn15] = (
       +rho*Y[jn]*rate_eval.n_n15__d_c14
       )

    jac[jt, jn] = (
       +rho*Y[jd]*rate_eval.n_d__t
       +rho*Y[jhe3]*rate_eval.n_he3__p_t
       +rho*Y[jhe4]*rate_eval.n_he4__d_t
       +rho*Y[jli6]*rate_eval.n_li6__he4_t
       +rho*Y[jbe9]*rate_eval.n_be9__t_li7
       +rho*Y[jb11]*rate_eval.n_b11__t_be9
       +2*5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       +rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2.50000000000000e-01*rho**3*2*Y[jn]*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       )

    jac[jt, jp] = (
       -rho*Y[jt]*rate_eval.p_t__he4
       -rho*Y[jt]*rate_eval.p_t__n_he3
       -rho*Y[jt]*rate_eval.p_t__d_d
       +rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       )

    jac[jt, jd] = (
       -rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jn]*rate_eval.n_d__t
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__p_t
       +rho*Y[jhe4]*rate_eval.d_he4__t_he3
       +rho*Y[jli8]*rate_eval.d_li8__t_li7
       )

    jac[jt, jt] = (
       -rate_eval.t__he3__weak__wc12
       -rate_eval.t__n_d
       -rho*Y[jp]*rate_eval.p_t__he4
       -rho*Y[jhe4]*rate_eval.he4_t__li7
       -rho*Y[jp]*rate_eval.p_t__n_he3
       -rho*Y[jp]*rate_eval.p_t__d_d
       -rho*Y[jd]*rate_eval.d_t__n_he4
       -rho*Y[jhe4]*rate_eval.he4_t__n_li6
       -rho*Y[jhe3]*rate_eval.t_he3__d_he4
       -rho*Y[jli7]*rate_eval.t_li7__n_be9
       -rho*Y[jli7]*rate_eval.t_li7__d_li8
       -rho*Y[jbe9]*rate_eval.t_be9__n_b11
       -2*5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       -rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       -rho*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       -rho*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jt, jhe3] = (
       -rho*Y[jt]*rate_eval.t_he3__d_he4
       -rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +rho*ye(Y)*rate_eval.he3__t__weak__electron_capture
       +rho*Y[jn]*rate_eval.n_he3__p_t
       )

    jac[jt, jhe4] = (
       -rho*Y[jt]*rate_eval.he4_t__li7
       -rho*Y[jt]*rate_eval.he4_t__n_li6
       +rate_eval.he4__p_t
       +rho*Y[jn]*rate_eval.n_he4__d_t
       +rho*Y[jd]*rate_eval.d_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       +rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       +2.50000000000000e-01*rho**3*Y[jn]**2*2*Y[jhe4]*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__t_be7
       )

    jac[jt, jli6] = (
       +rho*Y[jn]*rate_eval.n_li6__he4_t
       )

    jac[jt, jli7] = (
       -rho*Y[jt]*rate_eval.t_li7__n_be9
       -rho*Y[jt]*rate_eval.t_li7__d_li8
       -rho*Y[jt]*rate_eval.t_li7__n_n_he4_he4
       +rate_eval.li7__he4_t
       )

    jac[jt, jli8] = (
       +rho*Y[jd]*rate_eval.d_li8__t_li7
       )

    jac[jt, jbe7] = (
       -rho*Y[jt]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jt, jbe9] = (
       -rho*Y[jt]*rate_eval.t_be9__n_b11
       +rho*Y[jn]*rate_eval.n_be9__t_li7
       )

    jac[jt, jb11] = (
       +rho*Y[jn]*rate_eval.n_b11__t_be9
       )

    jac[jhe3, jn] = (
       -rho*Y[jhe3]*rate_eval.n_he3__he4
       -rho*Y[jhe3]*rate_eval.n_he3__p_t
       -rho*Y[jhe3]*rate_eval.n_he3__d_d
       +rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       )

    jac[jhe3, jp] = (
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.p_d__he3
       +rho*Y[jt]*rate_eval.p_t__n_he3
       +rho*Y[jhe4]*rate_eval.p_he4__d_he3
       +rho*Y[jli6]*rate_eval.p_li6__he4_he3
       +rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       +2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
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
       -rho*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jn]*rate_eval.n_he3__p_t
       -rho*Y[jn]*rate_eval.n_he3__d_d
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       -rho*Y[jt]*rate_eval.t_he3__d_he4
       -rho*Y[jhe4]*rate_eval.he4_he3__p_li6
       -rho*Y[jt]*rate_eval.t_he3__n_p_he4
       -2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       -rho*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       -rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe3, jhe4] = (
       -rho*Y[jhe3]*rate_eval.he4_he3__be7
       -rho*Y[jhe3]*rate_eval.he4_he3__p_li6
       +rate_eval.he4__n_he3
       +rho*Y[jp]*rate_eval.p_he4__d_he3
       +rho*Y[jd]*rate_eval.d_he4__t_he3
       +rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       +2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__he3_li7
       +2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_he4_he4__he3_be7
       )

    jac[jhe3, jli6] = (
       +rho*Y[jp]*rate_eval.p_li6__he4_he3
       )

    jac[jhe3, jli7] = (
       -rho*Y[jhe3]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jhe3, jbe7] = (
       -rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       +rate_eval.be7__he4_he3
       )

    jac[jhe4, jn] = (
       -rho*Y[jhe4]*rate_eval.n_he4__d_t
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       -5.00000000000000e-01*rho**2*2*Y[jn]*Y[jhe4]*rate_eval.n_n_he4__t_t
       -rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       -2*2.50000000000000e-01*rho**3*2*Y[jn]*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       -2*5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -2*5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -2*5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       +rho*Y[jhe3]*rate_eval.n_he3__he4
       +rho*Y[jli6]*rate_eval.n_li6__he4_t
       +2*rho*Y[jbe7]*rate_eval.n_be7__he4_he4
       +rho*Y[jb10]*rate_eval.n_b10__he4_li7
       +rho*Y[jb11]*rate_eval.n_b11__he4_li8
       +rho*Y[jc12]*rate_eval.n_c12__he4_be9
       +rho*Y[jn13]*rate_eval.n_n13__he4_b10
       +rho*Y[jn14]*rate_eval.n_n14__he4_b11
       +rho*Y[jn15]*rate_eval.n_n15__he4_b12
       +rho*Y[jo14]*rate_eval.n_o14__he4_c11
       +rho*Y[jo15]*rate_eval.n_o15__he4_c12
       +rho*Y[jo16]*rate_eval.n_o16__he4_c13
       +2*rho*Y[jb8]*rate_eval.n_b8__p_he4_he4
       +3*rho*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       )

    jac[jhe4, jp] = (
       -rho*Y[jhe4]*rate_eval.p_he4__d_he3
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__li6
       -rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_he4__he3_he3
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       +rho*Y[jt]*rate_eval.p_t__he4
       +rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jli6]*rate_eval.p_li6__he4_he3
       +2*rho*Y[jli7]*rate_eval.p_li7__he4_he4
       +rho*Y[jbe9]*rate_eval.p_be9__he4_li6
       +rho*Y[jb10]*rate_eval.p_b10__he4_be7
       +rho*Y[jb12]*rate_eval.p_b12__he4_be9
       +rho*Y[jc11]*rate_eval.p_c11__he4_b8
       +rho*Y[jc13]*rate_eval.p_c13__he4_b10
       +rho*Y[jc14]*rate_eval.p_c14__he4_b11
       +rho*Y[jn14]*rate_eval.p_n14__he4_c11
       +rho*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jo15]*rate_eval.p_o15__he4_n12
       +rho*Y[jo16]*rate_eval.p_o16__he4_n13
       +2*rho*Y[jli8]*rate_eval.p_li8__n_he4_he4
       +2*rho*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       +3*rho*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       +2*rho*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       )

    jac[jhe4, jd] = (
       -rho*Y[jhe4]*rate_eval.he4_d__li6
       -rho*Y[jhe4]*rate_eval.d_he4__t_he3
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       +rho*Y[jt]*rate_eval.d_t__n_he4
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*rho*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       )

    jac[jhe4, jt] = (
       -rho*Y[jhe4]*rate_eval.he4_t__li7
       -rho*Y[jhe4]*rate_eval.he4_t__n_li6
       +rho*Y[jp]*rate_eval.p_t__he4
       +rho*Y[jd]*rate_eval.d_t__n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__d_he4
       +5.00000000000000e-01*rho*2*Y[jt]*rate_eval.t_t__n_n_he4
       +rho*Y[jhe3]*rate_eval.t_he3__n_p_he4
       +2*rho*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +2*rho*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jhe4, jhe3] = (
       -rho*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jhe4]*rate_eval.he4_he3__p_li6
       +rho*Y[jn]*rate_eval.n_he3__he4
       +rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +rho*Y[jt]*rate_eval.t_he3__d_he4
       +rho*Y[jt]*rate_eval.t_he3__n_p_he4
       +5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       +2*rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe4, jhe4] = (
       -rate_eval.he4__n_he3
       -rate_eval.he4__p_t
       -rate_eval.he4__d_d
       -rho*Y[jd]*rate_eval.he4_d__li6
       -rho*Y[jt]*rate_eval.he4_t__li7
       -rho*Y[jhe3]*rate_eval.he4_he3__be7
       -rho*Y[jli6]*rate_eval.he4_li6__b10
       -rho*Y[jli7]*rate_eval.he4_li7__b11
       -rho*Y[jli8]*rate_eval.he4_li8__b12
       -rho*Y[jbe7]*rate_eval.he4_be7__c11
       -rho*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jt]*rate_eval.he4_t__n_li6
       -rho*Y[jhe3]*rate_eval.he4_he3__p_li6
       -rho*Y[jn]*rate_eval.n_he4__d_t
       -rho*Y[jp]*rate_eval.p_he4__d_he3
       -rho*Y[jd]*rate_eval.d_he4__t_he3
       -2*5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__n_be7
       -2*5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__p_li7
       -rho*Y[jli6]*rate_eval.he4_li6__p_be9
       -rho*Y[jli7]*rate_eval.he4_li7__n_b10
       -rho*Y[jli8]*rate_eval.he4_li8__n_b11
       -rho*Y[jbe7]*rate_eval.he4_be7__p_b10
       -rho*Y[jbe9]*rate_eval.he4_be9__n_c12
       -rho*Y[jbe9]*rate_eval.he4_be9__p_b12
       -rho*Y[jb8]*rate_eval.he4_b8__p_c11
       -rho*Y[jb10]*rate_eval.he4_b10__n_n13
       -rho*Y[jb10]*rate_eval.he4_b10__p_c13
       -rho*Y[jb11]*rate_eval.he4_b11__n_n14
       -rho*Y[jb11]*rate_eval.he4_b11__p_c14
       -rho*Y[jb12]*rate_eval.he4_b12__n_n15
       -rho*Y[jc11]*rate_eval.he4_c11__n_o14
       -rho*Y[jc11]*rate_eval.he4_c11__p_n14
       -rho*Y[jc12]*rate_eval.he4_c12__n_o15
       -rho*Y[jc12]*rate_eval.he4_c12__p_n15
       -rho*Y[jc13]*rate_eval.he4_c13__n_o16
       -rho*Y[jn12]*rate_eval.he4_n12__p_o15
       -rho*Y[jn13]*rate_eval.he4_n13__p_o16
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__li6
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__be9
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       -5.00000000000000e-01*rho**2*Y[jn]**2*rate_eval.n_n_he4__t_t
       -rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__t_he3
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_he4__he3_he3
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__p_li8
       -2*5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__d_li7
       -2*5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__n_b8
       -2*5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__d_be7
       -2*5.00000000000000e-01*rho**2*Y[jd]*2*Y[jhe4]*rate_eval.d_he4_he4__p_be9
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__n_c11
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__p_b11
       -2*2.50000000000000e-01*rho**3*Y[jn]**2*2*Y[jhe4]*rate_eval.n_n_he4_he4__t_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__he3_li7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__t_be7
       -2*5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__p_be9
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_he4_he4__he3_be7
       )

    jac[jhe4, jli6] = (
       -rho*Y[jhe4]*rate_eval.he4_li6__b10
       -rho*Y[jhe4]*rate_eval.he4_li6__p_be9
       +rate_eval.li6__he4_d
       +rate_eval.li6__n_p_he4
       +rho*Y[jn]*rate_eval.n_li6__he4_t
       +rho*Y[jp]*rate_eval.p_li6__he4_he3
       )

    jac[jhe4, jli7] = (
       -rho*Y[jhe4]*rate_eval.he4_li7__b11
       -rho*Y[jhe4]*rate_eval.he4_li7__n_b10
       +rate_eval.li7__he4_t
       +2*rho*Y[jp]*rate_eval.p_li7__he4_he4
       +2*rho*Y[jd]*rate_eval.d_li7__n_he4_he4
       +2*rho*Y[jt]*rate_eval.t_li7__n_n_he4_he4
       +2*rho*Y[jhe3]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jhe4, jli8] = (
       -rho*Y[jhe4]*rate_eval.he4_li8__b12
       -rho*Y[jhe4]*rate_eval.he4_li8__n_b11
       +2*rate_eval.li8__he4_he4__weak__wc12
       +2*rho*Y[jp]*rate_eval.p_li8__n_he4_he4
       )

    jac[jhe4, jbe7] = (
       -rho*Y[jhe4]*rate_eval.he4_be7__c11
       -rho*Y[jhe4]*rate_eval.he4_be7__p_b10
       +rate_eval.be7__he4_he3
       +2*rho*Y[jn]*rate_eval.n_be7__he4_he4
       +2*rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jt]*rate_eval.t_be7__n_p_he4_he4
       +2*rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe4, jbe9] = (
       -rho*Y[jhe4]*rate_eval.he4_be9__n_c12
       -rho*Y[jhe4]*rate_eval.he4_be9__p_b12
       +2*rate_eval.be9__n_he4_he4
       +rho*Y[jp]*rate_eval.p_be9__he4_li6
       +2*rho*Y[jp]*rate_eval.p_be9__d_he4_he4
       +2*rho*Y[jp]*rate_eval.p_be9__n_p_he4_he4
       )

    jac[jhe4, jb8] = (
       -rho*Y[jhe4]*rate_eval.he4_b8__p_c11
       +2*rate_eval.b8__he4_he4__weak__wc12
       +2*rho*Y[jn]*rate_eval.n_b8__p_he4_he4
       )

    jac[jhe4, jb10] = (
       -rho*Y[jhe4]*rate_eval.he4_b10__n_n13
       -rho*Y[jhe4]*rate_eval.he4_b10__p_c13
       +rate_eval.b10__he4_li6
       +rho*Y[jn]*rate_eval.n_b10__he4_li7
       +rho*Y[jp]*rate_eval.p_b10__he4_be7
       )

    jac[jhe4, jb11] = (
       -rho*Y[jhe4]*rate_eval.he4_b11__n_n14
       -rho*Y[jhe4]*rate_eval.he4_b11__p_c14
       +rate_eval.b11__he4_li7
       +rho*Y[jn]*rate_eval.n_b11__he4_li8
       +3*rho*Y[jp]*rate_eval.p_b11__he4_he4_he4
       )

    jac[jhe4, jb12] = (
       -rho*Y[jhe4]*rate_eval.he4_b12__n_n15
       +rate_eval.b12__he4_li8
       +rho*Y[jp]*rate_eval.p_b12__he4_be9
       )

    jac[jhe4, jc11] = (
       -rho*Y[jhe4]*rate_eval.he4_c11__n_o14
       -rho*Y[jhe4]*rate_eval.he4_c11__p_n14
       +rate_eval.c11__he4_be7
       +rho*Y[jp]*rate_eval.p_c11__he4_b8
       +3*rho*Y[jn]*rate_eval.n_c11__he4_he4_he4
       )

    jac[jhe4, jc12] = (
       -rho*Y[jhe4]*rate_eval.he4_c12__o16
       -rho*Y[jhe4]*rate_eval.he4_c12__n_o15
       -rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       +3*rate_eval.c12__he4_he4_he4
       +rho*Y[jn]*rate_eval.n_c12__he4_be9
       )

    jac[jhe4, jc13] = (
       -rho*Y[jhe4]*rate_eval.he4_c13__n_o16
       +rho*Y[jp]*rate_eval.p_c13__he4_b10
       )

    jac[jhe4, jc14] = (
       +rho*Y[jp]*rate_eval.p_c14__he4_b11
       )

    jac[jhe4, jn12] = (
       -rho*Y[jhe4]*rate_eval.he4_n12__p_o15
       )

    jac[jhe4, jn13] = (
       -rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       +rho*Y[jn]*rate_eval.n_n13__he4_b10
       )

    jac[jhe4, jn14] = (
       +rho*Y[jn]*rate_eval.n_n14__he4_b11
       +rho*Y[jp]*rate_eval.p_n14__he4_c11
       )

    jac[jhe4, jn15] = (
       +rho*Y[jn]*rate_eval.n_n15__he4_b12
       +rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jhe4, jo14] = (
       +rho*Y[jn]*rate_eval.n_o14__he4_c11
       )

    jac[jhe4, jo15] = (
       +rho*Y[jn]*rate_eval.n_o15__he4_c12
       +rho*Y[jp]*rate_eval.p_o15__he4_n12
       )

    jac[jhe4, jo16] = (
       +rate_eval.o16__he4_c12
       +rho*Y[jn]*rate_eval.n_o16__he4_c13
       +rho*Y[jp]*rate_eval.p_o16__he4_n13
       )

    jac[jli6, jn] = (
       -rho*Y[jli6]*rate_eval.n_li6__li7
       -rho*Y[jli6]*rate_eval.n_li6__he4_t
       +rho*Y[jbe7]*rate_eval.n_be7__d_li6
       +rho**2*Y[jp]*Y[jhe4]*rate_eval.n_p_he4__li6
       )

    jac[jli6, jp] = (
       -rho*Y[jli6]*rate_eval.p_li6__be7
       -rho*Y[jli6]*rate_eval.p_li6__he4_he3
       +rho*Y[jli7]*rate_eval.p_li7__d_li6
       +rho*Y[jbe9]*rate_eval.p_be9__he4_li6
       +rho**2*Y[jn]*Y[jhe4]*rate_eval.n_p_he4__li6
       )

    jac[jli6, jd] = (
       -rho*Y[jli6]*rate_eval.d_li6__n_be7
       -rho*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jhe4]*rate_eval.he4_d__li6
       )

    jac[jli6, jt] = (
       +rho*Y[jhe4]*rate_eval.he4_t__n_li6
       )

    jac[jli6, jhe3] = (
       +rho*Y[jhe4]*rate_eval.he4_he3__p_li6
       )

    jac[jli6, jhe4] = (
       -rho*Y[jli6]*rate_eval.he4_li6__b10
       -rho*Y[jli6]*rate_eval.he4_li6__p_be9
       +rho*Y[jd]*rate_eval.he4_d__li6
       +rho*Y[jt]*rate_eval.he4_t__n_li6
       +rho*Y[jhe3]*rate_eval.he4_he3__p_li6
       +rho**2*Y[jn]*Y[jp]*rate_eval.n_p_he4__li6
       )

    jac[jli6, jli6] = (
       -rate_eval.li6__he4_d
       -rate_eval.li6__n_p_he4
       -rho*Y[jn]*rate_eval.n_li6__li7
       -rho*Y[jp]*rate_eval.p_li6__be7
       -rho*Y[jhe4]*rate_eval.he4_li6__b10
       -rho*Y[jn]*rate_eval.n_li6__he4_t
       -rho*Y[jp]*rate_eval.p_li6__he4_he3
       -rho*Y[jd]*rate_eval.d_li6__n_be7
       -rho*Y[jd]*rate_eval.d_li6__p_li7
       -rho*Y[jhe4]*rate_eval.he4_li6__p_be9
       )

    jac[jli6, jli7] = (
       +rate_eval.li7__n_li6
       +rho*Y[jp]*rate_eval.p_li7__d_li6
       )

    jac[jli6, jbe7] = (
       +rate_eval.be7__p_li6
       +rho*Y[jn]*rate_eval.n_be7__d_li6
       )

    jac[jli6, jbe9] = (
       +rho*Y[jp]*rate_eval.p_be9__he4_li6
       )

    jac[jli6, jb10] = (
       +rate_eval.b10__he4_li6
       )

    jac[jli7, jn] = (
       -rho*Y[jli7]*rate_eval.n_li7__li8
       +rho*Y[jli6]*rate_eval.n_li6__li7
       +rho*Y[jbe7]*rate_eval.n_be7__p_li7
       +rho*Y[jbe9]*rate_eval.n_be9__t_li7
       +rho*Y[jb10]*rate_eval.n_b10__he4_li7
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__d_li7
       +2.50000000000000e-01*rho**3*2*Y[jn]*Y[jhe4]**2*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       )

    jac[jli7, jp] = (
       -rho*Y[jli7]*rate_eval.p_li7__n_be7
       -rho*Y[jli7]*rate_eval.p_li7__d_li6
       -rho*Y[jli7]*rate_eval.p_li7__he4_he4
       +rho*Y[jli8]*rate_eval.p_li8__d_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__he3_li7
       )

    jac[jli7, jd] = (
       -rho*Y[jli7]*rate_eval.d_li7__p_li8
       -rho*Y[jli7]*rate_eval.d_li7__n_he4_he4
       +rho*Y[jli6]*rate_eval.d_li6__p_li7
       +rho*Y[jli8]*rate_eval.d_li8__t_li7
       )

    jac[jli7, jt] = (
       -rho*Y[jli7]*rate_eval.t_li7__n_be9
       -rho*Y[jli7]*rate_eval.t_li7__d_li8
       -rho*Y[jli7]*rate_eval.t_li7__n_n_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_t__li7
       )

    jac[jli7, jhe3] = (
       -rho*Y[jli7]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jli7, jhe4] = (
       -rho*Y[jli7]*rate_eval.he4_li7__b11
       -rho*Y[jli7]*rate_eval.he4_li7__n_b10
       +rho*Y[jt]*rate_eval.he4_t__li7
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__p_li7
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__d_li7
       +2.50000000000000e-01*rho**3*Y[jn]**2*2*Y[jhe4]*rate_eval.n_n_he4_he4__t_li7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__he3_li7
       )

    jac[jli7, jli6] = (
       +rho*Y[jn]*rate_eval.n_li6__li7
       +rho*Y[jd]*rate_eval.d_li6__p_li7
       )

    jac[jli7, jli7] = (
       -rate_eval.li7__n_li6
       -rate_eval.li7__he4_t
       -rho*Y[jn]*rate_eval.n_li7__li8
       -rho*Y[jhe4]*rate_eval.he4_li7__b11
       -rho*Y[jp]*rate_eval.p_li7__n_be7
       -rho*Y[jp]*rate_eval.p_li7__d_li6
       -rho*Y[jp]*rate_eval.p_li7__he4_he4
       -rho*Y[jd]*rate_eval.d_li7__p_li8
       -rho*Y[jt]*rate_eval.t_li7__n_be9
       -rho*Y[jt]*rate_eval.t_li7__d_li8
       -rho*Y[jhe4]*rate_eval.he4_li7__n_b10
       -rho*Y[jd]*rate_eval.d_li7__n_he4_he4
       -rho*Y[jt]*rate_eval.t_li7__n_n_he4_he4
       -rho*Y[jhe3]*rate_eval.he3_li7__n_p_he4_he4
       )

    jac[jli7, jli8] = (
       +rate_eval.li8__n_li7
       +rho*Y[jp]*rate_eval.p_li8__d_li7
       +rho*Y[jd]*rate_eval.d_li8__t_li7
       )

    jac[jli7, jbe7] = (
       +rho*ye(Y)*rate_eval.be7__li7__weak__electron_capture
       +rho*Y[jn]*rate_eval.n_be7__p_li7
       )

    jac[jli7, jbe9] = (
       +rho*Y[jn]*rate_eval.n_be9__t_li7
       )

    jac[jli7, jb10] = (
       +rho*Y[jn]*rate_eval.n_b10__he4_li7
       )

    jac[jli7, jb11] = (
       +rate_eval.b11__he4_li7
       )

    jac[jli8, jn] = (
       +rho*Y[jli7]*rate_eval.n_li7__li8
       +rho*Y[jbe9]*rate_eval.n_be9__d_li8
       +rho*Y[jb11]*rate_eval.n_b11__he4_li8
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__p_li8
       )

    jac[jli8, jp] = (
       -rho*Y[jli8]*rate_eval.p_li8__d_li7
       -rho*Y[jli8]*rate_eval.p_li8__n_he4_he4
       )

    jac[jli8, jd] = (
       -rho*Y[jli8]*rate_eval.d_li8__n_be9
       -rho*Y[jli8]*rate_eval.d_li8__t_li7
       +rho*Y[jli7]*rate_eval.d_li7__p_li8
       )

    jac[jli8, jt] = (
       +rho*Y[jli7]*rate_eval.t_li7__d_li8
       )

    jac[jli8, jhe4] = (
       -rho*Y[jli8]*rate_eval.he4_li8__b12
       -rho*Y[jli8]*rate_eval.he4_li8__n_b11
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__p_li8
       )

    jac[jli8, jli7] = (
       +rho*Y[jn]*rate_eval.n_li7__li8
       +rho*Y[jd]*rate_eval.d_li7__p_li8
       +rho*Y[jt]*rate_eval.t_li7__d_li8
       )

    jac[jli8, jli8] = (
       -rate_eval.li8__n_li7
       -rate_eval.li8__he4_he4__weak__wc12
       -rho*Y[jhe4]*rate_eval.he4_li8__b12
       -rho*Y[jp]*rate_eval.p_li8__d_li7
       -rho*Y[jd]*rate_eval.d_li8__n_be9
       -rho*Y[jd]*rate_eval.d_li8__t_li7
       -rho*Y[jhe4]*rate_eval.he4_li8__n_b11
       -rho*Y[jp]*rate_eval.p_li8__n_he4_he4
       )

    jac[jli8, jbe9] = (
       +rho*Y[jn]*rate_eval.n_be9__d_li8
       )

    jac[jli8, jb11] = (
       +rho*Y[jn]*rate_eval.n_b11__he4_li8
       )

    jac[jli8, jb12] = (
       +rate_eval.b12__he4_li8
       )

    jac[jbe7, jn] = (
       -rho*Y[jbe7]*rate_eval.n_be7__p_li7
       -rho*Y[jbe7]*rate_eval.n_be7__d_li6
       -rho*Y[jbe7]*rate_eval.n_be7__he4_he4
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       )

    jac[jbe7, jp] = (
       -rho*Y[jbe7]*rate_eval.p_be7__b8
       +rho*Y[jli6]*rate_eval.p_li6__be7
       +rho*Y[jli7]*rate_eval.p_li7__n_be7
       +rho*Y[jb10]*rate_eval.p_b10__he4_be7
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__d_be7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__t_be7
       +2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_he4_he4__he3_be7
       )

    jac[jbe7, jd] = (
       -rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +rho*Y[jli6]*rate_eval.d_li6__n_be7
       )

    jac[jbe7, jt] = (
       -rho*Y[jbe7]*rate_eval.t_be7__n_p_he4_he4
       )

    jac[jbe7, jhe3] = (
       -rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_he3__be7
       )

    jac[jbe7, jhe4] = (
       -rho*Y[jbe7]*rate_eval.he4_be7__c11
       -rho*Y[jbe7]*rate_eval.he4_be7__p_b10
       +rho*Y[jhe3]*rate_eval.he4_he3__be7
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.he4_he4__n_be7
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__d_be7
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__t_be7
       +2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_he4_he4__he3_be7
       )

    jac[jbe7, jli6] = (
       +rho*Y[jp]*rate_eval.p_li6__be7
       +rho*Y[jd]*rate_eval.d_li6__n_be7
       )

    jac[jbe7, jli7] = (
       +rho*Y[jp]*rate_eval.p_li7__n_be7
       )

    jac[jbe7, jbe7] = (
       -rho*ye(Y)*rate_eval.be7__li7__weak__electron_capture
       -rate_eval.be7__p_li6
       -rate_eval.be7__he4_he3
       -rho*Y[jp]*rate_eval.p_be7__b8
       -rho*Y[jhe4]*rate_eval.he4_be7__c11
       -rho*Y[jn]*rate_eval.n_be7__p_li7
       -rho*Y[jn]*rate_eval.n_be7__d_li6
       -rho*Y[jn]*rate_eval.n_be7__he4_he4
       -rho*Y[jhe4]*rate_eval.he4_be7__p_b10
       -rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       -rho*Y[jt]*rate_eval.t_be7__n_p_he4_he4
       -rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jbe7, jb8] = (
       +rate_eval.b8__p_be7
       )

    jac[jbe7, jb10] = (
       +rho*Y[jp]*rate_eval.p_b10__he4_be7
       )

    jac[jbe7, jc11] = (
       +rate_eval.c11__he4_be7
       )

    jac[jbe9, jn] = (
       -rho*Y[jbe9]*rate_eval.n_be9__d_li8
       -rho*Y[jbe9]*rate_eval.n_be9__t_li7
       +rho*Y[jb11]*rate_eval.n_b11__t_be9
       +rho*Y[jc12]*rate_eval.n_c12__he4_be9
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.n_he4_he4__be9
       +5.00000000000000e-01*rho**3*Y[jp]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       )

    jac[jbe9, jp] = (
       -rho*Y[jbe9]*rate_eval.p_be9__b10
       -rho*Y[jbe9]*rate_eval.p_be9__he4_li6
       -rho*Y[jbe9]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jbe9]*rate_eval.p_be9__n_p_he4_he4
       +rho*Y[jb12]*rate_eval.p_b12__he4_be9
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jhe4]**2*rate_eval.n_p_he4_he4__p_be9
       )

    jac[jbe9, jd] = (
       +rho*Y[jli8]*rate_eval.d_li8__n_be9
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.d_he4_he4__p_be9
       )

    jac[jbe9, jt] = (
       -rho*Y[jbe9]*rate_eval.t_be9__n_b11
       +rho*Y[jli7]*rate_eval.t_li7__n_be9
       )

    jac[jbe9, jhe4] = (
       -rho*Y[jbe9]*rate_eval.he4_be9__n_c12
       -rho*Y[jbe9]*rate_eval.he4_be9__p_b12
       +rho*Y[jli6]*rate_eval.he4_li6__p_be9
       +5.00000000000000e-01*rho**2*Y[jn]*2*Y[jhe4]*rate_eval.n_he4_he4__be9
       +5.00000000000000e-01*rho**2*Y[jd]*2*Y[jhe4]*rate_eval.d_he4_he4__p_be9
       +5.00000000000000e-01*rho**3*Y[jn]*Y[jp]*2*Y[jhe4]*rate_eval.n_p_he4_he4__p_be9
       )

    jac[jbe9, jli6] = (
       +rho*Y[jhe4]*rate_eval.he4_li6__p_be9
       )

    jac[jbe9, jli7] = (
       +rho*Y[jt]*rate_eval.t_li7__n_be9
       )

    jac[jbe9, jli8] = (
       +rho*Y[jd]*rate_eval.d_li8__n_be9
       )

    jac[jbe9, jbe9] = (
       -rate_eval.be9__n_he4_he4
       -rho*Y[jp]*rate_eval.p_be9__b10
       -rho*Y[jn]*rate_eval.n_be9__d_li8
       -rho*Y[jn]*rate_eval.n_be9__t_li7
       -rho*Y[jp]*rate_eval.p_be9__he4_li6
       -rho*Y[jt]*rate_eval.t_be9__n_b11
       -rho*Y[jhe4]*rate_eval.he4_be9__n_c12
       -rho*Y[jhe4]*rate_eval.he4_be9__p_b12
       -rho*Y[jp]*rate_eval.p_be9__d_he4_he4
       -rho*Y[jp]*rate_eval.p_be9__n_p_he4_he4
       )

    jac[jbe9, jb10] = (
       +rate_eval.b10__p_be9
       )

    jac[jbe9, jb11] = (
       +rho*Y[jn]*rate_eval.n_b11__t_be9
       )

    jac[jbe9, jb12] = (
       +rho*Y[jp]*rate_eval.p_b12__he4_be9
       )

    jac[jbe9, jc12] = (
       +rho*Y[jn]*rate_eval.n_c12__he4_be9
       )

    jac[jb8, jn] = (
       -rho*Y[jb8]*rate_eval.n_b8__p_he4_he4
       )

    jac[jb8, jp] = (
       +rho*Y[jbe7]*rate_eval.p_be7__b8
       +rho*Y[jc11]*rate_eval.p_c11__he4_b8
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_he4_he4__n_b8
       )

    jac[jb8, jhe4] = (
       -rho*Y[jb8]*rate_eval.he4_b8__p_c11
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_he4_he4__n_b8
       )

    jac[jb8, jbe7] = (
       +rho*Y[jp]*rate_eval.p_be7__b8
       )

    jac[jb8, jb8] = (
       -rate_eval.b8__p_be7
       -rate_eval.b8__he4_he4__weak__wc12
       -rho*Y[jhe4]*rate_eval.he4_b8__p_c11
       -rho*Y[jn]*rate_eval.n_b8__p_he4_he4
       )

    jac[jb8, jc11] = (
       +rho*Y[jp]*rate_eval.p_c11__he4_b8
       )

    jac[jb10, jn] = (
       -rho*Y[jb10]*rate_eval.n_b10__b11
       -rho*Y[jb10]*rate_eval.n_b10__he4_li7
       +rho*Y[jn13]*rate_eval.n_n13__he4_b10
       )

    jac[jb10, jp] = (
       -rho*Y[jb10]*rate_eval.p_b10__c11
       -rho*Y[jb10]*rate_eval.p_b10__he4_be7
       +rho*Y[jbe9]*rate_eval.p_be9__b10
       +rho*Y[jc13]*rate_eval.p_c13__he4_b10
       )

    jac[jb10, jhe4] = (
       -rho*Y[jb10]*rate_eval.he4_b10__n_n13
       -rho*Y[jb10]*rate_eval.he4_b10__p_c13
       +rho*Y[jli6]*rate_eval.he4_li6__b10
       +rho*Y[jli7]*rate_eval.he4_li7__n_b10
       +rho*Y[jbe7]*rate_eval.he4_be7__p_b10
       )

    jac[jb10, jli6] = (
       +rho*Y[jhe4]*rate_eval.he4_li6__b10
       )

    jac[jb10, jli7] = (
       +rho*Y[jhe4]*rate_eval.he4_li7__n_b10
       )

    jac[jb10, jbe7] = (
       +rho*Y[jhe4]*rate_eval.he4_be7__p_b10
       )

    jac[jb10, jbe9] = (
       +rho*Y[jp]*rate_eval.p_be9__b10
       )

    jac[jb10, jb10] = (
       -rate_eval.b10__p_be9
       -rate_eval.b10__he4_li6
       -rho*Y[jn]*rate_eval.n_b10__b11
       -rho*Y[jp]*rate_eval.p_b10__c11
       -rho*Y[jn]*rate_eval.n_b10__he4_li7
       -rho*Y[jp]*rate_eval.p_b10__he4_be7
       -rho*Y[jhe4]*rate_eval.he4_b10__n_n13
       -rho*Y[jhe4]*rate_eval.he4_b10__p_c13
       )

    jac[jb10, jb11] = (
       +rate_eval.b11__n_b10
       )

    jac[jb10, jc11] = (
       +rate_eval.c11__p_b10
       )

    jac[jb10, jc13] = (
       +rho*Y[jp]*rate_eval.p_c13__he4_b10
       )

    jac[jb10, jn13] = (
       +rho*Y[jn]*rate_eval.n_n13__he4_b10
       )

    jac[jb11, jn] = (
       -rho*Y[jb11]*rate_eval.n_b11__b12
       -rho*Y[jb11]*rate_eval.n_b11__t_be9
       -rho*Y[jb11]*rate_eval.n_b11__he4_li8
       +rho*Y[jb10]*rate_eval.n_b10__b11
       +rho*Y[jc11]*rate_eval.n_c11__p_b11
       +rho*Y[jn14]*rate_eval.n_n14__he4_b11
       )

    jac[jb11, jp] = (
       -rho*Y[jb11]*rate_eval.p_b11__c12
       -rho*Y[jb11]*rate_eval.p_b11__n_c11
       -rho*Y[jb11]*rate_eval.p_b11__he4_he4_he4
       +rho*Y[jc14]*rate_eval.p_c14__he4_b11
       )

    jac[jb11, jt] = (
       +rho*Y[jbe9]*rate_eval.t_be9__n_b11
       )

    jac[jb11, jhe4] = (
       -rho*Y[jb11]*rate_eval.he4_b11__n_n14
       -rho*Y[jb11]*rate_eval.he4_b11__p_c14
       +rho*Y[jli7]*rate_eval.he4_li7__b11
       +rho*Y[jli8]*rate_eval.he4_li8__n_b11
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__p_b11
       )

    jac[jb11, jli7] = (
       +rho*Y[jhe4]*rate_eval.he4_li7__b11
       )

    jac[jb11, jli8] = (
       +rho*Y[jhe4]*rate_eval.he4_li8__n_b11
       )

    jac[jb11, jbe9] = (
       +rho*Y[jt]*rate_eval.t_be9__n_b11
       )

    jac[jb11, jb10] = (
       +rho*Y[jn]*rate_eval.n_b10__b11
       )

    jac[jb11, jb11] = (
       -rate_eval.b11__n_b10
       -rate_eval.b11__he4_li7
       -rho*Y[jn]*rate_eval.n_b11__b12
       -rho*Y[jp]*rate_eval.p_b11__c12
       -rho*Y[jn]*rate_eval.n_b11__t_be9
       -rho*Y[jn]*rate_eval.n_b11__he4_li8
       -rho*Y[jp]*rate_eval.p_b11__n_c11
       -rho*Y[jhe4]*rate_eval.he4_b11__n_n14
       -rho*Y[jhe4]*rate_eval.he4_b11__p_c14
       -rho*Y[jp]*rate_eval.p_b11__he4_he4_he4
       )

    jac[jb11, jb12] = (
       +rate_eval.b12__n_b11
       )

    jac[jb11, jc11] = (
       +rate_eval.c11__b11__weak__wc12
       +rho*Y[jn]*rate_eval.n_c11__p_b11
       )

    jac[jb11, jc12] = (
       +rate_eval.c12__p_b11
       )

    jac[jb11, jc14] = (
       +rho*Y[jp]*rate_eval.p_c14__he4_b11
       )

    jac[jb11, jn14] = (
       +rho*Y[jn]*rate_eval.n_n14__he4_b11
       )

    jac[jb12, jn] = (
       +rho*Y[jb11]*rate_eval.n_b11__b12
       +rho*Y[jc12]*rate_eval.n_c12__p_b12
       +rho*Y[jn15]*rate_eval.n_n15__he4_b12
       )

    jac[jb12, jp] = (
       -rho*Y[jb12]*rate_eval.p_b12__n_c12
       -rho*Y[jb12]*rate_eval.p_b12__he4_be9
       )

    jac[jb12, jhe4] = (
       -rho*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jli8]*rate_eval.he4_li8__b12
       +rho*Y[jbe9]*rate_eval.he4_be9__p_b12
       )

    jac[jb12, jli8] = (
       +rho*Y[jhe4]*rate_eval.he4_li8__b12
       )

    jac[jb12, jbe9] = (
       +rho*Y[jhe4]*rate_eval.he4_be9__p_b12
       )

    jac[jb12, jb11] = (
       +rho*Y[jn]*rate_eval.n_b11__b12
       )

    jac[jb12, jb12] = (
       -rate_eval.b12__c12__weak__wc17
       -rate_eval.b12__n_b11
       -rate_eval.b12__he4_li8
       -rho*Y[jp]*rate_eval.p_b12__n_c12
       -rho*Y[jp]*rate_eval.p_b12__he4_be9
       -rho*Y[jhe4]*rate_eval.he4_b12__n_n15
       )

    jac[jb12, jc12] = (
       +rho*Y[jn]*rate_eval.n_c12__p_b12
       )

    jac[jb12, jn15] = (
       +rho*Y[jn]*rate_eval.n_n15__he4_b12
       )

    jac[jc11, jn] = (
       -rho*Y[jc11]*rate_eval.n_c11__c12
       -rho*Y[jc11]*rate_eval.n_c11__p_b11
       -rho*Y[jc11]*rate_eval.n_c11__he4_he4_he4
       +rho*Y[jo14]*rate_eval.n_o14__he4_c11
       )

    jac[jc11, jp] = (
       -rho*Y[jc11]*rate_eval.p_c11__n12
       -rho*Y[jc11]*rate_eval.p_c11__he4_b8
       +rho*Y[jb10]*rate_eval.p_b10__c11
       +rho*Y[jb11]*rate_eval.p_b11__n_c11
       +rho*Y[jn14]*rate_eval.p_n14__he4_c11
       )

    jac[jc11, jhe4] = (
       -rho*Y[jc11]*rate_eval.he4_c11__n_o14
       -rho*Y[jc11]*rate_eval.he4_c11__p_n14
       +rho*Y[jbe7]*rate_eval.he4_be7__c11
       +rho*Y[jb8]*rate_eval.he4_b8__p_c11
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__n_c11
       )

    jac[jc11, jbe7] = (
       +rho*Y[jhe4]*rate_eval.he4_be7__c11
       )

    jac[jc11, jb8] = (
       +rho*Y[jhe4]*rate_eval.he4_b8__p_c11
       )

    jac[jc11, jb10] = (
       +rho*Y[jp]*rate_eval.p_b10__c11
       )

    jac[jc11, jb11] = (
       +rho*Y[jp]*rate_eval.p_b11__n_c11
       )

    jac[jc11, jc11] = (
       -rate_eval.c11__b11__weak__wc12
       -rate_eval.c11__p_b10
       -rate_eval.c11__he4_be7
       -rho*Y[jn]*rate_eval.n_c11__c12
       -rho*Y[jp]*rate_eval.p_c11__n12
       -rho*Y[jn]*rate_eval.n_c11__p_b11
       -rho*Y[jp]*rate_eval.p_c11__he4_b8
       -rho*Y[jhe4]*rate_eval.he4_c11__n_o14
       -rho*Y[jhe4]*rate_eval.he4_c11__p_n14
       -rho*Y[jn]*rate_eval.n_c11__he4_he4_he4
       )

    jac[jc11, jc12] = (
       +rate_eval.c12__n_c11
       )

    jac[jc11, jn12] = (
       +rate_eval.n12__p_c11
       )

    jac[jc11, jn14] = (
       +rho*Y[jp]*rate_eval.p_n14__he4_c11
       )

    jac[jc11, jo14] = (
       +rho*Y[jn]*rate_eval.n_o14__he4_c11
       )

    jac[jc12, jn] = (
       -rho*Y[jc12]*rate_eval.n_c12__c13
       -rho*Y[jc12]*rate_eval.n_c12__p_b12
       -rho*Y[jc12]*rate_eval.n_c12__he4_be9
       +rho*Y[jc11]*rate_eval.n_c11__c12
       +rho*Y[jo15]*rate_eval.n_o15__he4_c12
       )

    jac[jc12, jp] = (
       -rho*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jb11]*rate_eval.p_b11__c12
       +rho*Y[jb12]*rate_eval.p_b12__n_c12
       +rho*Y[jn15]*rate_eval.p_n15__he4_c12
       )

    jac[jc12, jhe4] = (
       -rho*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jc12]*rate_eval.he4_c12__n_o15
       -rho*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jbe9]*rate_eval.he4_be9__n_c12
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       )

    jac[jc12, jbe9] = (
       +rho*Y[jhe4]*rate_eval.he4_be9__n_c12
       )

    jac[jc12, jb11] = (
       +rho*Y[jp]*rate_eval.p_b11__c12
       )

    jac[jc12, jb12] = (
       +rate_eval.b12__c12__weak__wc17
       +rho*Y[jp]*rate_eval.p_b12__n_c12
       )

    jac[jc12, jc11] = (
       +rho*Y[jn]*rate_eval.n_c11__c12
       )

    jac[jc12, jc12] = (
       -rate_eval.c12__n_c11
       -rate_eval.c12__p_b11
       -rate_eval.c12__he4_he4_he4
       -rho*Y[jn]*rate_eval.n_c12__c13
       -rho*Y[jp]*rate_eval.p_c12__n13
       -rho*Y[jhe4]*rate_eval.he4_c12__o16
       -rho*Y[jn]*rate_eval.n_c12__p_b12
       -rho*Y[jn]*rate_eval.n_c12__he4_be9
       -rho*Y[jhe4]*rate_eval.he4_c12__n_o15
       -rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       )

    jac[jc12, jc13] = (
       +rate_eval.c13__n_c12
       )

    jac[jc12, jn12] = (
       +rate_eval.n12__c12__weak__wc12
       )

    jac[jc12, jn13] = (
       +rate_eval.n13__p_c12
       )

    jac[jc12, jn15] = (
       +rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jc12, jo15] = (
       +rho*Y[jn]*rate_eval.n_o15__he4_c12
       )

    jac[jc12, jo16] = (
       +rate_eval.o16__he4_c12
       )

    jac[jc13, jn] = (
       -rho*Y[jc13]*rate_eval.n_c13__c14
       +rho*Y[jc12]*rate_eval.n_c12__c13
       +rho*Y[jn13]*rate_eval.n_n13__p_c13
       +rho*Y[jn14]*rate_eval.n_n14__d_c13
       +rho*Y[jo16]*rate_eval.n_o16__he4_c13
       )

    jac[jc13, jp] = (
       -rho*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jc13]*rate_eval.p_c13__n_n13
       -rho*Y[jc13]*rate_eval.p_c13__he4_b10
       )

    jac[jc13, jd] = (
       -rho*Y[jc13]*rate_eval.d_c13__n_n14
       )

    jac[jc13, jhe4] = (
       -rho*Y[jc13]*rate_eval.he4_c13__n_o16
       +rho*Y[jb10]*rate_eval.he4_b10__p_c13
       )

    jac[jc13, jb10] = (
       +rho*Y[jhe4]*rate_eval.he4_b10__p_c13
       )

    jac[jc13, jc12] = (
       +rho*Y[jn]*rate_eval.n_c12__c13
       )

    jac[jc13, jc13] = (
       -rate_eval.c13__n_c12
       -rho*Y[jn]*rate_eval.n_c13__c14
       -rho*Y[jp]*rate_eval.p_c13__n14
       -rho*Y[jp]*rate_eval.p_c13__n_n13
       -rho*Y[jp]*rate_eval.p_c13__he4_b10
       -rho*Y[jd]*rate_eval.d_c13__n_n14
       -rho*Y[jhe4]*rate_eval.he4_c13__n_o16
       )

    jac[jc13, jc14] = (
       +rate_eval.c14__n_c13
       )

    jac[jc13, jn13] = (
       +rate_eval.n13__c13__weak__wc12
       +rho*Y[jn]*rate_eval.n_n13__p_c13
       )

    jac[jc13, jn14] = (
       +rate_eval.n14__p_c13
       +rho*Y[jn]*rate_eval.n_n14__d_c13
       )

    jac[jc13, jo16] = (
       +rho*Y[jn]*rate_eval.n_o16__he4_c13
       )

    jac[jc14, jn] = (
       +rho*Y[jc13]*rate_eval.n_c13__c14
       +rho*Y[jn14]*rate_eval.n_n14__p_c14
       +rho*Y[jn15]*rate_eval.n_n15__d_c14
       )

    jac[jc14, jp] = (
       -rho*Y[jc14]*rate_eval.p_c14__n15
       -rho*Y[jc14]*rate_eval.p_c14__n_n14
       -rho*Y[jc14]*rate_eval.p_c14__he4_b11
       )

    jac[jc14, jd] = (
       -rho*Y[jc14]*rate_eval.d_c14__n_n15
       )

    jac[jc14, jhe4] = (
       +rho*Y[jb11]*rate_eval.he4_b11__p_c14
       )

    jac[jc14, jb11] = (
       +rho*Y[jhe4]*rate_eval.he4_b11__p_c14
       )

    jac[jc14, jc13] = (
       +rho*Y[jn]*rate_eval.n_c13__c14
       )

    jac[jc14, jc14] = (
       -rate_eval.c14__n14__weak__wc12
       -rate_eval.c14__n_c13
       -rho*Y[jp]*rate_eval.p_c14__n15
       -rho*Y[jp]*rate_eval.p_c14__n_n14
       -rho*Y[jp]*rate_eval.p_c14__he4_b11
       -rho*Y[jd]*rate_eval.d_c14__n_n15
       )

    jac[jc14, jn14] = (
       +rho*Y[jn]*rate_eval.n_n14__p_c14
       )

    jac[jc14, jn15] = (
       +rate_eval.n15__p_c14
       +rho*Y[jn]*rate_eval.n_n15__d_c14
       )

    jac[jn12, jp] = (
       +rho*Y[jc11]*rate_eval.p_c11__n12
       +rho*Y[jo15]*rate_eval.p_o15__he4_n12
       )

    jac[jn12, jhe4] = (
       -rho*Y[jn12]*rate_eval.he4_n12__p_o15
       )

    jac[jn12, jc11] = (
       +rho*Y[jp]*rate_eval.p_c11__n12
       )

    jac[jn12, jn12] = (
       -rate_eval.n12__c12__weak__wc12
       -rate_eval.n12__p_c11
       -rho*Y[jhe4]*rate_eval.he4_n12__p_o15
       )

    jac[jn12, jo15] = (
       +rho*Y[jp]*rate_eval.p_o15__he4_n12
       )

    jac[jn13, jn] = (
       -rho*Y[jn13]*rate_eval.n_n13__n14
       -rho*Y[jn13]*rate_eval.n_n13__p_c13
       -rho*Y[jn13]*rate_eval.n_n13__he4_b10
       )

    jac[jn13, jp] = (
       -rho*Y[jn13]*rate_eval.p_n13__o14
       +rho*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jc13]*rate_eval.p_c13__n_n13
       +rho*Y[jo16]*rate_eval.p_o16__he4_n13
       )

    jac[jn13, jhe4] = (
       -rho*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jb10]*rate_eval.he4_b10__n_n13
       )

    jac[jn13, jb10] = (
       +rho*Y[jhe4]*rate_eval.he4_b10__n_n13
       )

    jac[jn13, jc12] = (
       +rho*Y[jp]*rate_eval.p_c12__n13
       )

    jac[jn13, jc13] = (
       +rho*Y[jp]*rate_eval.p_c13__n_n13
       )

    jac[jn13, jn13] = (
       -rate_eval.n13__c13__weak__wc12
       -rate_eval.n13__p_c12
       -rho*Y[jn]*rate_eval.n_n13__n14
       -rho*Y[jp]*rate_eval.p_n13__o14
       -rho*Y[jn]*rate_eval.n_n13__p_c13
       -rho*Y[jn]*rate_eval.n_n13__he4_b10
       -rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jn13, jn14] = (
       +rate_eval.n14__n_n13
       )

    jac[jn13, jo14] = (
       +rate_eval.o14__p_n13
       )

    jac[jn13, jo16] = (
       +rho*Y[jp]*rate_eval.p_o16__he4_n13
       )

    jac[jn14, jn] = (
       -rho*Y[jn14]*rate_eval.n_n14__n15
       -rho*Y[jn14]*rate_eval.n_n14__p_c14
       -rho*Y[jn14]*rate_eval.n_n14__d_c13
       -rho*Y[jn14]*rate_eval.n_n14__he4_b11
       +rho*Y[jn13]*rate_eval.n_n13__n14
       +rho*Y[jo14]*rate_eval.n_o14__p_n14
       )

    jac[jn14, jp] = (
       -rho*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jn14]*rate_eval.p_n14__n_o14
       -rho*Y[jn14]*rate_eval.p_n14__he4_c11
       +rho*Y[jc13]*rate_eval.p_c13__n14
       +rho*Y[jc14]*rate_eval.p_c14__n_n14
       )

    jac[jn14, jd] = (
       +rho*Y[jc13]*rate_eval.d_c13__n_n14
       )

    jac[jn14, jhe4] = (
       +rho*Y[jb11]*rate_eval.he4_b11__n_n14
       +rho*Y[jc11]*rate_eval.he4_c11__p_n14
       )

    jac[jn14, jb11] = (
       +rho*Y[jhe4]*rate_eval.he4_b11__n_n14
       )

    jac[jn14, jc11] = (
       +rho*Y[jhe4]*rate_eval.he4_c11__p_n14
       )

    jac[jn14, jc13] = (
       +rho*Y[jp]*rate_eval.p_c13__n14
       +rho*Y[jd]*rate_eval.d_c13__n_n14
       )

    jac[jn14, jc14] = (
       +rate_eval.c14__n14__weak__wc12
       +rho*Y[jp]*rate_eval.p_c14__n_n14
       )

    jac[jn14, jn13] = (
       +rho*Y[jn]*rate_eval.n_n13__n14
       )

    jac[jn14, jn14] = (
       -rate_eval.n14__n_n13
       -rate_eval.n14__p_c13
       -rho*Y[jn]*rate_eval.n_n14__n15
       -rho*Y[jp]*rate_eval.p_n14__o15
       -rho*Y[jn]*rate_eval.n_n14__p_c14
       -rho*Y[jn]*rate_eval.n_n14__d_c13
       -rho*Y[jn]*rate_eval.n_n14__he4_b11
       -rho*Y[jp]*rate_eval.p_n14__n_o14
       -rho*Y[jp]*rate_eval.p_n14__he4_c11
       )

    jac[jn14, jn15] = (
       +rate_eval.n15__n_n14
       )

    jac[jn14, jo14] = (
       +rate_eval.o14__n14__weak__wc12
       +rho*Y[jn]*rate_eval.n_o14__p_n14
       )

    jac[jn14, jo15] = (
       +rate_eval.o15__p_n14
       )

    jac[jn15, jn] = (
       -rho*Y[jn15]*rate_eval.n_n15__d_c14
       -rho*Y[jn15]*rate_eval.n_n15__he4_b12
       +rho*Y[jn14]*rate_eval.n_n14__n15
       +rho*Y[jo15]*rate_eval.n_o15__p_n15
       )

    jac[jn15, jp] = (
       -rho*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jn15]*rate_eval.p_n15__n_o15
       -rho*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jc14]*rate_eval.p_c14__n15
       )

    jac[jn15, jd] = (
       +rho*Y[jc14]*rate_eval.d_c14__n_n15
       )

    jac[jn15, jhe4] = (
       +rho*Y[jb12]*rate_eval.he4_b12__n_n15
       +rho*Y[jc12]*rate_eval.he4_c12__p_n15
       )

    jac[jn15, jb12] = (
       +rho*Y[jhe4]*rate_eval.he4_b12__n_n15
       )

    jac[jn15, jc12] = (
       +rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       )

    jac[jn15, jc14] = (
       +rho*Y[jp]*rate_eval.p_c14__n15
       +rho*Y[jd]*rate_eval.d_c14__n_n15
       )

    jac[jn15, jn14] = (
       +rho*Y[jn]*rate_eval.n_n14__n15
       )

    jac[jn15, jn15] = (
       -rate_eval.n15__n_n14
       -rate_eval.n15__p_c14
       -rho*Y[jp]*rate_eval.p_n15__o16
       -rho*Y[jn]*rate_eval.n_n15__d_c14
       -rho*Y[jn]*rate_eval.n_n15__he4_b12
       -rho*Y[jp]*rate_eval.p_n15__n_o15
       -rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jn15, jo15] = (
       +rate_eval.o15__n15__weak__wc12
       +rho*Y[jn]*rate_eval.n_o15__p_n15
       )

    jac[jn15, jo16] = (
       +rate_eval.o16__p_n15
       )

    jac[jo14, jn] = (
       -rho*Y[jo14]*rate_eval.n_o14__o15
       -rho*Y[jo14]*rate_eval.n_o14__p_n14
       -rho*Y[jo14]*rate_eval.n_o14__he4_c11
       )

    jac[jo14, jp] = (
       +rho*Y[jn13]*rate_eval.p_n13__o14
       +rho*Y[jn14]*rate_eval.p_n14__n_o14
       )

    jac[jo14, jhe4] = (
       +rho*Y[jc11]*rate_eval.he4_c11__n_o14
       )

    jac[jo14, jc11] = (
       +rho*Y[jhe4]*rate_eval.he4_c11__n_o14
       )

    jac[jo14, jn13] = (
       +rho*Y[jp]*rate_eval.p_n13__o14
       )

    jac[jo14, jn14] = (
       +rho*Y[jp]*rate_eval.p_n14__n_o14
       )

    jac[jo14, jo14] = (
       -rate_eval.o14__n14__weak__wc12
       -rate_eval.o14__p_n13
       -rho*Y[jn]*rate_eval.n_o14__o15
       -rho*Y[jn]*rate_eval.n_o14__p_n14
       -rho*Y[jn]*rate_eval.n_o14__he4_c11
       )

    jac[jo14, jo15] = (
       +rate_eval.o15__n_o14
       )

    jac[jo15, jn] = (
       -rho*Y[jo15]*rate_eval.n_o15__o16
       -rho*Y[jo15]*rate_eval.n_o15__p_n15
       -rho*Y[jo15]*rate_eval.n_o15__he4_c12
       +rho*Y[jo14]*rate_eval.n_o14__o15
       )

    jac[jo15, jp] = (
       -rho*Y[jo15]*rate_eval.p_o15__he4_n12
       +rho*Y[jn14]*rate_eval.p_n14__o15
       +rho*Y[jn15]*rate_eval.p_n15__n_o15
       )

    jac[jo15, jhe4] = (
       +rho*Y[jc12]*rate_eval.he4_c12__n_o15
       +rho*Y[jn12]*rate_eval.he4_n12__p_o15
       )

    jac[jo15, jc12] = (
       +rho*Y[jhe4]*rate_eval.he4_c12__n_o15
       )

    jac[jo15, jn12] = (
       +rho*Y[jhe4]*rate_eval.he4_n12__p_o15
       )

    jac[jo15, jn14] = (
       +rho*Y[jp]*rate_eval.p_n14__o15
       )

    jac[jo15, jn15] = (
       +rho*Y[jp]*rate_eval.p_n15__n_o15
       )

    jac[jo15, jo14] = (
       +rho*Y[jn]*rate_eval.n_o14__o15
       )

    jac[jo15, jo15] = (
       -rate_eval.o15__n15__weak__wc12
       -rate_eval.o15__n_o14
       -rate_eval.o15__p_n14
       -rho*Y[jn]*rate_eval.n_o15__o16
       -rho*Y[jn]*rate_eval.n_o15__p_n15
       -rho*Y[jn]*rate_eval.n_o15__he4_c12
       -rho*Y[jp]*rate_eval.p_o15__he4_n12
       )

    jac[jo15, jo16] = (
       +rate_eval.o16__n_o15
       )

    jac[jo16, jn] = (
       -rho*Y[jo16]*rate_eval.n_o16__he4_c13
       +rho*Y[jo15]*rate_eval.n_o15__o16
       )

    jac[jo16, jp] = (
       -rho*Y[jo16]*rate_eval.p_o16__he4_n13
       +rho*Y[jn15]*rate_eval.p_n15__o16
       )

    jac[jo16, jhe4] = (
       +rho*Y[jc12]*rate_eval.he4_c12__o16
       +rho*Y[jc13]*rate_eval.he4_c13__n_o16
       +rho*Y[jn13]*rate_eval.he4_n13__p_o16
       )

    jac[jo16, jc12] = (
       +rho*Y[jhe4]*rate_eval.he4_c12__o16
       )

    jac[jo16, jc13] = (
       +rho*Y[jhe4]*rate_eval.he4_c13__n_o16
       )

    jac[jo16, jn13] = (
       +rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jo16, jn15] = (
       +rho*Y[jp]*rate_eval.p_n15__o16
       )

    jac[jo16, jo15] = (
       +rho*Y[jn]*rate_eval.n_o15__o16
       )

    jac[jo16, jo16] = (
       -rate_eval.o16__n_o15
       -rate_eval.o16__p_n15
       -rate_eval.o16__he4_c12
       -rho*Y[jn]*rate_eval.n_o16__he4_c13
       -rho*Y[jp]*rate_eval.p_o16__he4_n13
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


   if __name__ == "__main__":
      cc.compile()
