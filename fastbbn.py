import numpy as np
import scipy.linalg as la
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt
import AoT_net as bbn_n
nNucs=bbn_n.nnuc()
import full_AoT_net as bbn_full
fnNucs=bbn_full.nnuc()


#Background
#Special functions og deres afledte
def L(z):
    return special.kn(2,z)/z

def M(z):
    return (3/4*special.kn(3,z)+1/4*special.kn(1,z))/z

def dMdz(z):
    return -3/z**2*special.kn(3,z) -special.kn(2,z)/z

def N(z):
    return (1/2*special.kn(4,z)+1/2*special.kn(2,z))/z


timeunit =1.519*10**21  #MeV/hbar in unit of 1/second 

n_life=879.6*timeunit   #Neutron lifetime in units of MeV/hbar
Q=1.293                 #neutron proton mass difference in MeV
M_u=931.494102          #atomic mass unit in MeV
SBC=np.pi**2/60         #Stefan-Boltzmann constant in natural units
G=6.709e-45             #gravatational constant in units of c=hbar=MeV=1
#infapprox=1e3           #large number acting as upper limit on itegrals
n_nu=3.046              #number of neutrino families with correction from 
                        #Nollett and Steigman, BBN and the CMB constrain neutrino coupled light WIMPs, 2015

TMeV2T9=11.60451812 #conversion factor from MeV to 10^9K
cm3s=1.167*10**-11  #conversion factor for cm^3/s
gcm3=232012         #conversion factor for g/cm^3
meter=5.068e12      #conversion factor for m
cm=5.068e10         #conversion factor for cm
barn=389.4          #conversion factor for barn
e_mass=0.51099895   #electron mass in MeV


def rho_e(T): #electron/positron density, chemical potential assumed to be 0 so cosh(phi*n)=1
    z=e_mass/T
    return 2/np.pi**2*e_mass**4*np.sum([(-1)**(n+1)*M(n*z) for n in range(1,10)])

def drho_e(T): #derivative of rho_e with respect to temperature
    z=e_mass/T
    return 2/np.pi**2*e_mass**4*np.sum([(-1)**(n)*n*z/T*dMdz(n*z) for n in range(1,10)])

def P_e(T): #electron/positron pressure
    z=e_mass/T
    return 2/np.pi**2*e_mass**4*np.sum([(-1)**(n+1)/(n*z)*L(n*z) for n in range(1,10)])  

def rho_gamma(T):   #photon energy density
    return (np.pi**2)/15*T**4 

def drho_gamma(T):  #derivative
    return 4*(np.pi**2)/15*T**3 

def P_gamma(T):     #photon pressure
    return rho_gamma(T)/3



### Initial conditions ###

T_ini=27/TMeV2T9            #initial temperature in MeV

t_ini=1.226*10**21/T_ini**2 #initial time in hbar/MeV
t_cut=1*timeunit            #time at which full reaction network is added

z_ini=e_mass/T_ini

n_gamma_ini= 1.20206*2/np.pi**2*T_ini**3    #initial number density of photons based on theory
n_gamma_ini= rho_gamma(T_ini)/(2.701*T_ini) #initial number density of photons based mean photon energy

eta=6.1e-10     #CMB baryon-to-photon ratio
eta_ini=eta*(1+(rho_e(T_ini)+P_e(T_ini))/(rho_gamma(T_ini)+P_gamma(T_ini))) #entropi bevarelse

rho_nu_ini=n_nu*7/8*(np.pi**2)/15*T_ini**4 #initial neutrino density

rho_b_ini=M_u*eta_ini*n_gamma_ini    #initial baryon density

rho_tot_ini=rho_e(T_ini)+rho_gamma(T_ini)+rho_nu_ini#+rho_b_ini  #initial total energy density

H_ini=np.sqrt(8*np.pi/3*G*rho_tot_ini)

def rho_nu(T,a): #neutrino energy density
    #return 7/8*np.pi**2/15*T**4
    return rho_nu_ini/a**4

def rho_b(a):
    return rho_b_ini/a**3


### Solving the background ###

def rho_tot(T,a): #total density
    return rho_e(T)+rho_gamma(T)+rho_nu(T,a)#+rho_b(a)

def rho_set(T,a): #total density of non-decoupled components
    return rho_e(T)+rho_gamma(T)#+rho_b(a)

def H(T,a):   #Hubble parameter as given by Friedmann eq, ignoring cosmological constant
    return np.sqrt(8*np.pi/3*G*rho_tot(T,a))

#Derivative from Kavano D.18
def dTdt(t,T,a):
    return -3*H(T,a)/((drho_e(T)+drho_gamma(T))/(rho_set(T,a) + P_e(T)+P_gamma(T)))

#Derivative from how H is defined
def dadt(t,T,a):
    return a*H(T,a)


t_range=[t_ini,5e4*timeunit]            #time range for integration
#t_space=np.linspace(*t_range,1000)      #time range for approximate temperature

#combining derivatives
def dbackground(t,y):   #solve h and T, y[0] = T, y[1] = a
    return [dTdt(t,*y),dadt(t,*y)]

n_bparams=2


#Setup isotopes based on Alterbbn
Y_labels=['n','p','H2','H3','He3','He4','Li6','Li7','Li8','Be7','Be9','B8','B10','B11','B12','C11','C12','C13','C14','N12','N13','N14','N15','O14','O15','O16']
Alter_Yl=["n","p","H2","H3","He3","He4","Li6","Li7","Be7","Li8","B8","Be9","B10","B11","C11","B12","C12","N12","C13","N13","C14","N14","O14","N15","O15","O16"]
A=np.array([1,1,2,3,3,4,6,7,8,7,9,8,10,11,12,11,12,13,14,12,13,14,15,14,15,16])
Z=sorted([0,1,1,1,2,2,3,3,4,3,5,4,5,5,6,5,6,7,6,7,6,7,8,7,8,8.])
Alter_A=[1.,1.,2.,3.,3.,4.,6.,7.,7.,8.,8.,9.,10.,11.,11.,12.,12.,12.,13.,13.,14.,14.,14.,15.,15.,16.]
Alter_Z=[0.,1.,1.,1.,2.,2.,3.,3.,4.,3.,5.,4.,5.,5.,6.,5.,6.,7.,6.,7.,6.,7.,8.,7.,8.,8.]
Alterspin=[0.5,0.5,1.,0.5,0.5,0.,1.,1.5,1.5,2.,2.,1.5,3.,1.5,1.5,1.,0.,1.,0.5,0.5,0.,1.,0.,0.5,0.5,0.]
Alter_mass_excess=[8.071388,7.289028,13.135825,14.949915,14.931325,2.424931,14.0864,14.9078,15.7696,20.9464,22.9212,11.34758,12.05086,8.6680,10.6506,13.3690,0.,17.3382,3.125036,5.3455,3.019916,2.863440,8.006521,0.101439,2.8554,-4.737036]
Alter_mass=[Alter_A[i]*M_u+Alter_mass_excess[i] for i in range(fnNucs)]

def Altersort(L):
    if type(L)==list:
        return [x for (a,x) in sorted(zip(zip(A,Z),L), key=lambda pair: pair[0])]
    else:
        return np.array([x for (a,x) in sorted(zip(zip(A,Z),L), key=lambda pair: pair[0])])
#print([label for _, label in sorted(zip(A, Y_labels))])

def PNAsort(L):
    if type(L)==list:
        return [x for (a,x) in sorted(zip(zip(Alter_Z,Alter_A),L), key=lambda pair: pair[0])]
    else:
        return np.array([x for (a,x) in sorted(zip(zip(Alter_Z,Alter_A),L), key=lambda pair: pair[0])])    


#Initial conditions for Y

spin=PNAsort(Alterspin)
A=np.array(PNAsort(Alter_A))
m_Nucs = np.array(PNAsort(Alter_mass))
B=[(m_Nucs[1]*Z[i]+m_Nucs[0]*(A[i]-Z[i]))-m_Nucs[i] for i in range(fnNucs)]
g = 1+2*np.array(spin)

#Determine baryon density in cgs, for use in network
def rho_bY_cgs(y):
    return sum(m_Nucs[:len(y[n_bparams:])]*y[n_bparams:])*eta_ini*n_gamma_ini/y[1]**3*gcm3

#Guess initial conditions from thermal equilibrium

def get_Y_ini(Xn_ini):
    Y_ini2 = np.zeros(nNucs)+1e-50
        
    for iter1 in range(30):
        Xp_ini = np.exp(Q/T_ini)*Xn_ini
        Y_ini2[0] = Xn_ini        #Set initial neutron mass fraction
        Y_ini2[1] = Xp_ini
        #Set initial proton mass fraction
        X_sum = Y_ini2[0] + Y_ini2[1]
        for i in range(2,nNucs):
            tmp = special.zeta(3)**(A[i] - 1)*np.pi**((1 - A[i])/2)*2**((3*A[i] - 5)/2)*A[i]**(5/2)

            Y_ini2[i] = g[i]*tmp*(T_ini/m_Nucs[0])**(3*(A[i] - 1)/2)*eta_ini**(A[i] - 1)*Xp_ini**Z[i]*Xn_ini**(A[i] - Z[i])*np.exp(B[i]/T_ini)/A[i]
            X_sum += Y_ini2[i]*A[i]
        #print(X_sum)
        Xn_ini += (1 - X_sum)/nNucs
        
    return Y_ini2
Xn_ini = 1/(np.exp(Q/T_ini)+1)
Y_ini2 = get_Y_ini(Xn_ini)


#Refine initial conditions from Jacobian
AdYdt_ini=lambda Y : Altersort(bbn_n.rhs(t_ini/timeunit, PNAsort(Y) ,rho_bY_cgs([T_ini,1]+list(PNAsort(Y))), T_ini*TMeV2T9*1e9))
AdYdt_jac=lambda Y : bbn_n.jacobian(t_ini/timeunit, PNAsort(Y) ,rho_bY_cgs([T_ini,1]+list(PNAsort(Y))), T_ini*TMeV2T9*1e9)[:, Altersort(range(nNucs))][Altersort(range(nNucs))]

aY_ini=Altersort(Y_ini2)
def solve_using_svd(U, s, Vh, b):
    bb = U.T @ b
    y = bb/s
    x = Vh.T @ y
    return x
Yj = np.array([YY for YY in aY_ini])
cut_start = 0
for cut in range(cut_start, len(Yj) - 2, 1):
    for j in range(10):
        fyj = -AdYdt_ini(Yj)
        jac = AdYdt_jac(Yj)
        if np.any(np.isnan(Yj)) or np.any(np.isinf(Yj)):
            print('Yj:', Yj)
            raise ValueError
        if np.any(np.isnan(fyj)) or np.any(np.isinf(fyj)):
            print('Yj:', Yj)
            print('fyj:', fyj)
            raise ValueError
        # Implement cut:
        fyj = fyj[cut:]
        jac = jac[cut:, cut:]
        
        # Solution using SVD
        U, s, Vh = la.svd(jac)    
        x = solve_using_svd(U, s, Vh, fyj)
        for k in range(1):
            #A · δx = A · (x + δx) − b
            residuals = jac @ x - fyj
            dx = solve_using_svd(U, s, Vh, residuals)
            x -= dx
        Yj[cut:] += x


#Combining background and network

initial_param=[T_ini,1]+list(PNAsort(Yj))
n_params=nNucs+n_bparams

def ndall(t,y):             
    return dbackground(t,y[:n_bparams])+list( bbn_n.rhs(t/timeunit, y[n_bparams:],rho_bY_cgs(y), y[0]*TMeV2T9*1e9)/timeunit)



#solving for early times and high temperature

def jacY_anal(t,y):
    return bbn_n.jacobian(t/timeunit, y[n_bparams:],rho_bY_cgs(y), y[0]*TMeV2T9*1e9)/timeunit

def jacfun(t,x):
    return np.array([ndall(t,row)  for row in x.T]).T

def jacobian(t,y):
    jac=np.append(np.zeros((nNucs,n_bparams)),jacY_anal(t,y),axis=1)
    return np.append(np.zeros((n_bparams,n_params)),jac,axis=0)

jacsolY = integrate.solve_ivp(ndall, [0,t_cut-t_ini], initial_param,method='Radau',atol=1e-80,rtol=1e-6,jac=jacobian)#,t_eval=t_space)


#guess initial conditons for heavy nuclei from thermal equilibrium
abun=[abun[-1] for abun in jacsolY.y[n_bparams:]]

T_cut=jacsolY.y[0][-1]

dYdt_cut=lambda Y : bbn_full.rhs(t_cut/timeunit, Y ,rho_bY_cgs([param[-1] for param in jacsolY.y]), T_cut*TMeV2T9*1e9)

Y_cut=np.zeros(fnNucs)+1e-80
Y_cut[:nNucs] = abun

def get_Y_cut(abun_cut):
    Y_ini3 = np.zeros(fnNucs)+1e-80
        
    for iter1 in range(30):
        
        Xn_cut = abun_cut[0]
        Xp_cut = abun_cut[1]
        Y_ini3[:nNucs] = abun_cut
        for i in range(nNucs,fnNucs):
            tmp = special.zeta(3)**(A[i] - 1)*np.pi**((1 - A[i])/2)*2**((3*A[i] - 5)/2)*A[i]**(5/2)

            Y_ini3[i] = g[i]*tmp*(T_ini/m_Nucs[0])**(3*(A[i] - 1)/2)*eta_ini**(A[i] - 1)*Xp_cut**Z[i]*Xn_cut**(A[i] - Z[i])*np.exp(B[i]/T_cut)/A[i]

    return Y_ini3

Y_cut2 = get_Y_cut(abun)

#refine initial conditons for heavy nuclei from Jacobian
AdYdt_cut=lambda Y : Altersort(bbn_full.rhs(t_cut/timeunit, PNAsort(Y) ,rho_bY_cgs([param[-1] for param in jacsolY.y]), T_cut*TMeV2T9*1e9))
AdYdt_jac_cut=lambda Y : bbn_full.jacobian(t_cut/timeunit, PNAsort(Y) ,rho_bY_cgs([param[-1] for param in jacsolY.y]), T_cut*TMeV2T9*1e9)[:, Altersort(range(fnNucs))][Altersort(range(fnNucs))]
import scipy.linalg as la
aY_cut=Altersort(Y_cut2)
Yj = np.array([YY for YY in aY_cut])
cut_start = nNucs
for cut in range(cut_start, len(Yj) - 2, 1):
    for j in range(10):

        fyj = -AdYdt_cut(Yj)
        jac = AdYdt_jac_cut(Yj)
        if np.any(np.isnan(Yj)) or np.any(np.isinf(Yj)):
            print('Yj:', Yj)
            raise ValueError
        if np.any(np.isnan(fyj)) or np.any(np.isinf(fyj)):
            print('Yj:', Yj)
            print('fyj:', fyj)
            raise ValueError
        # Implement cut:
        fyj = fyj[cut:]
        jac = jac[cut:, cut:]
        
        # Solution using SVD
        U, s, Vh = la.svd(jac)    
        x = solve_using_svd(U, s, Vh, fyj)
        for k in range(1):
            #A · δx = A · (x + δx) − b
            residuals = jac @ x - fyj
            dx = solve_using_svd(U, s, Vh, residuals)
            x -= dx
        Yj[cut:] += x

#Combining background and full network

cut_param=[param[-1] for param in jacsolY.y[:n_bparams]]+list(PNAsort(Yj))
t_range_cut=[0,1e5*timeunit-t_cut+t_ini]
full_n_params=fnNucs+n_bparams

def ndfull(t,y):                
    return dbackground(t,y[:n_bparams])+list( bbn_full.rhs(t/timeunit, y[n_bparams:],rho_bY_cgs(y), y[0]*TMeV2T9*1e9)/timeunit)

#solving with full network
def full_jac(t,y):
    return bbn_full.jacobian(t/timeunit, y[n_bparams:],rho_bY_cgs(y), y[0]*TMeV2T9*1e9)/timeunit

def full_jacobian(t,y):
    jac=np.append(np.zeros((fnNucs,n_bparams)),full_jac(t,y),axis=1)
    return np.append(np.zeros((n_bparams,full_n_params)),jac,axis=0)

fullsolY = integrate.solve_ivp(ndfull, t_range_cut, cut_param,method='Radau',atol=1e-80,rtol=1e-6,jac=full_jacobian)#,t_eval=t_space)

#combine early and late solutions
solY=np.concatenate((np.concatenate((jacsolY.y,np.multiply(np.ones((len(jacsolY.y[0]),fnNucs-nNucs)),cut_param[n_params:]).T)),fullsolY.y),axis=1)
soltime=np.concatenate((jacsolY.t+t_ini,fullsolY.t+t_cut))/timeunit

# Plot the results    
plt.figure('abundance',figsize=(6.4, 8))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors*=3
line=['-']*10+['--']*10+[':']*10
for i in range(fnNucs):
    plt.plot(soltime, A[i]*solY[n_bparams+i],line[i], color=colors[i], label=Y_labels[i])

plt.xlabel('Time in seconds')
plt.ylabel('Mass fraction')
plt.ylim(1e-20,3)
plt.xlim(t_ini/timeunit,1e5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('abundance.png')
plt.show()

#Print final abundances
abun=[fabun[-1] for fabun in fullsolY.y[n_bparams:]]
print('\t Yp  '+'\t\t H2/H '+'\t\t H3/H '+'\t\t Li7/H '+'\t\t Li6/H '+'\t\t Be7/H ')
print('value:\t '"{:.3e}".format(4*abun[5])+'\t '+"{:.3e}".format(abun[2]/abun[1])+'\t '+"{:.3e}".format((abun[3]+abun[4])/abun[1])+'\t '+"{:.3e}".format((abun[7]+abun[9])/abun[1])+'\t '+"{:.3e}".format((abun[6])/abun[1])+'\t '+"{:.3e}".format((abun[9])/abun[1]))
