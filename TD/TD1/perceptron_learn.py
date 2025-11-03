import numpy as np
import matplotlib . pyplot as plt
import math , csv

inputValue = np. array ([[0 ,0] , [0 ,1] , [1 ,0] , [1 ,1]]) 
numIn = 4
desired_out = np. array ([0 ,0 ,0 ,1]) # Logical And function

bias = -1
coeff = 0.7 #(epsilon coefficient)

weights = -1*2.* np. random . rand (3 ,1) # initialization of synaptic weights

iterations = 1000
error = np. ones ((1 , iterations ))

for i in range (0, iterations ):
    out = np. zeros ((4 ,1))
    for j in range (0, numIn ):
        y = bias * weights[0]+ inputValue[j ][0]* weights [1]+ inputValue[j][1]*weights [2];
        out [j] = 1/(1+ math .exp(-y));
        delta = desired_out [j]-out[j];
        weights [0] = weights [0]+ coeff * bias * delta ;
        weights [1] = weights [1]+ coeff * inputValue [j ][0]* delta ;
        weights [2] = weights [2]+ coeff * inputValue [j ][1]* delta ;

    error [0][ i]= delta ;
    
plt.plot (np. arange (0, iterations , 1) , error [0])
plt.ylabel ('Error ')
plt.xlabel (" Iterations Number ")
plt.show ()

print ("w0 : "+ str( weights [0][0]) +"\nw1 : "+ str( weights [1][0]) +"\nw2 : "+ str (
weights [2][0]) +"\n")

# exportation
with open ('weights .csv ', 'w') as out_file :
    writer = csv . writer ( out_file )
    writer . writerow (( 'W0 ', 'W1 ', 'W2 '))
    writer . writerow (( str ( weights [0][0]) , str( weights [1][0]) , str ( weights
[2][0]) ))
print (" Values file exported ")