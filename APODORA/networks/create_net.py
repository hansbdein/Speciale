import pynucastro as pyna
from APODORA.networks.rates import p__n, n__p
import os



def BBN_net(nuclei):
    '''create pynucastro network with modified rates for n__p and p__n'''
    
    reaclibrary = pyna.ReacLibLibrary()
    bbn_library = reaclibrary.linking_nuclei(nuclei)

    bbn_library += pyna.Library(rates=[p__n,n__p])
    bbn_library.remove_rate(bbn_library.get_rate('n__p__weak__wc12'))
    return pyna.networks.PythonNetwork(libraries=bbn_library)


def write_AoTnetwork(net,networkname):
    net.write_network(networkname)

    file=open(networkname, 'a')
    file.write('''
    #For AoT compilation of the network
    def AoT(networkname):
               
        """function to compile the network Ahead of Time"""
               
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
    ''') # Write some text
    file.close() # Close the file

    current_directory = os.getcwd()
    print(f'Network saved in {current_directory}')
