print('Hyggehejsa')#husk at give permission chmod u+x alter_eta.x

import subprocess 

# Run the compiled program with an input parameter
input_param = "3e-10"
result = subprocess.run(["/home/hansbdein/Speciale/alterbbn_v2.2/alter_eta.x", input_param], capture_output=True)

# Omega_b = 4.171*10^-31 g/cm^3 
# svarer til Eta =6.1e-10
# Print the output of the program
#print(result.stdout.decode("utf-8"))
for info in result.stdout.decode("utf-8").split('\n')[6:11]:
    print(info)
    #print('lmao')
