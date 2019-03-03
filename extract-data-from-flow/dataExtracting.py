'''
This is an old version of data extracting from flow.
new version is to be added
'''
import numpy as np
#---------------------------read data from openfoam files----------------------
#fileNames = ['I1', 'I2', '4I3', 'I4', 'I5', 'I6', 'nuLES']
fileNames = ['I1', 'I2', 'I3', 'I4', 'I5', 'nuLES']
internalField = list()
for name in fileNames:
    with open(name, 'r') as f:
        lines = f.read().replace('\n', ' ')
        lines = lines.split('(')[1].split(')')[0].strip(' ')
        linesList = lines.split(' ')
        internalField.append(np.array(linesList))
internalField = np.array(internalField)
internalField = internalField.T
#---------------------------format output to flie 'instances'------------------
with open('instances', 'w') as f:
    for name in fileNames:
        f.write(12*' ' + name + ',')
    f.write('\n')
    for row in internalField:
        for feature in row:
            f.write('%15.10f,' % float(feature))
        f.write('\n')
#---------------------------generate flie 'instances.arff'---------------------
with open('instances.arff', 'w') as f:
    f.write('@relation turbulence_viscosity\n')
    for name in fileNames:
        f.write("@attribute '" + name + "' numeric\n")
    f.write('@data\n')
    for row in internalField:
        for feature in row:
            f.write(feature)
            if feature != row[-1]:
                f.write(',')          
        f.write('\n')