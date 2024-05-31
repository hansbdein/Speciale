import pynucastro as pyna
""""custom """

p__n_string = '''
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

'''

n__p_string = '''
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

'''

class n__p_Rate(pyna.Rate):
    def __init__(self, reactants=None, products=None,
                 r0=1.0, T0=1.0, nu=0):

        # we'll take the Q value just to be the change in binding energy
        Q = 0
        for n in reactants:
            Q += -n.A * n.nucbind
        for n in products:
            Q += n.A * n.nucbind

        # we set the chapter to custom so the network knows how to deal with it
        self.chapter = "custom"
        self.fname = "n__p"
        self.reverse = None
    
        # call the Rate init to do the remaining initialization
        super().__init__(reactants=reactants, products=products, Q=Q)

        self.r0 = r0
        self.T0 = T0
        self.nu = nu

    def function_string_py(self):
        """return a string containing a python function that computes
        the rate"""
        return n__p_string

    def eval(self, T, rhoY=None):
        return self.r0 * (T / self.T0)**self.nu
    
n__p = n__p_Rate(reactants=[pyna.Nucleus("n")],
                  products=[pyna.Nucleus("p")],
                  r0=1, T0=1, nu=1)


class p__n_Rate(pyna.Rate):
    def __init__(self, reactants=None, products=None,
                 r0=1.0, T0=1.0, nu=0):

        # we'll take the Q value just to be the change in binding energy
        Q = 0
        for n in reactants:
            Q += -n.A * n.nucbind
        for n in products:
            Q += n.A * n.nucbind

        # we set the chapter to custom so the network knows how to deal with it
        self.chapter = "custom"
        self.fname = 'p__n'

        self.reverse = None
    
        # call the Rate init to do the remaining initialization
        super().__init__(reactants=reactants, products=products, Q=Q)

        self.r0 = r0
        self.T0 = T0
        self.nu = nu

    def function_string_py(self):
        """return a string containing a python function that computes
        the rate"""
        return p__n_string

    def eval(self, T, rhoY=None):
        return self.r0 * (T / self.T0)**self.nu
    

p__n = p__n_Rate(reactants=[pyna.Nucleus("p")],
                  products=[pyna.Nucleus("n")],
                  r0=1, T0=1, nu=1)
