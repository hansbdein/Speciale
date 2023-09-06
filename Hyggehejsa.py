print('Hyggehejsa')#husk at give permission chmod u+x alter_eta.x

import subprocess 

# Run the compiled program with an input parameter

input_param = "6.1e-10"
result = subprocess.run(["/home/hansbdein/Speciale/alterbbn_v2.2/alter_eta.x", input_param], capture_output=True)

# Omega_b = 4.171*10^-31 g/cm^3 
# svarer til Eta =6.1e-10
# Print the output of the program
#print(result.stdout.decode("utf-8"))
for info in result.stdout.decode("utf-8").split('\n')[6:11]:
    print(info)
    #print('lmao')

'''
	// Declare a file pointer
    FILE *file;

    // Open a file for writing (create if it doesn't exist, truncate if it does)
    file = fopen("output.txt", "w");

    // Check if the file was opened successfully
    if (file == NULL) {
        printf("Error opening the file.\n");
        return 1; // Return an error code
    }

    // Print data to the file
    fprintf(file, "This is a line of text.\n");
    fprintf(file, "Another line of text.\n");
	fprintf(file, "The number is: %lf\n", Ti);

    // Close the file when you're done
    fclose(file);
'''