#!/usr/bin/python3
import numpy as np
import math
from pandas import read_csv
import os
import time
import subprocess

# -------------------------
def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    return list(bin(n).replace("0b", ""))

# -------------------------
def mcci_format(alpha,beta):
    line = ""
    alpha_array = decimalToBinary(alpha)
    beta_array = decimalToBinary(beta)

    indx1 = 0
    indx2 = 0
    for j1 in reversed(range(len(alpha_array))):
        if (alpha_array[j1] == '1'):
            line += str(indx1+1)+" "
        indx1 += 1
    for j2 in reversed(range(len(beta_array))):
        if (beta_array[j2] == '1'):
            line += (str(indx2+1)+" ")
        indx2 += 1
    line += '\n'
    return line

# -------------------------
def comma(deci):
    line = []
    array = decimalToBinary(deci)

    indx = 0
    for j in reversed(range(len(array))):
        if (array[j] == '1'):
            line.append(indx+1)
        indx += 1
    return line

# -------------------------
def mcci_in(p1,p2):
    lines = fmcci.readlines()
    fmcci.seek(0)
    lines[2] = "n_up            = "+str(nelec)+"                    ! number of up electrons\n"
    
    line = "mo_up           = "
    alpha_array = comma(p1)
    for i in range(nelec-1):
        line += str(alpha_array[i])+","
    line += str(alpha_array[nelec-1])
    line += " ! occupied spin up orbital labels\n"
    lines[3] = line

    lines[4] = "n_dn            = "+str(nelec)+"\n"
    line = "mo_dn           = "
    beta_array = comma(p2)
    for i in range(nelec-1):
        line += str(beta_array[i])+","
    line += str(beta_array[nelec-1])
    line += " ! occupied spin up orbital labels\n"
    lines[5] = line
    for i in range(len(lines)):
        fmcci.write(str(lines[i]))

    fmcci.truncate()

# -------------------------
fgen=open("general_input.in")
lines=fgen.readlines()
for i in range(len(lines)):
    lines[i]=lines[i].partition("#")[0]
    lines[i]=lines[i].rstrip()
#dcsf = int(lines[0])
nelec = int(lines[0])
dspace = int(lines[1]) 

fgen.close()
# -------------------------
fmcci = open("mcci.in","r+")
# -------------------------
# 0 det number (actual); 1-2: alpha beta decimal equivalence; 3: Predicted ci
dPred = np.loadtxt("Pred.out",usecols=(1,2),dtype=int)
Pred_ci = np.loadtxt("Pred.out",usecols=(3),dtype=float)
#print(dPred[0],Pred_ci[0])  # check if properly uploaded
Pred_ci = abs(Pred_ci)
# -------------------------

indx = range(len(dPred))
zipped_pair = zip(Pred_ci,indx)
sortedz= sorted(zipped_pair,key=lambda tup:tup[0],reverse=True)
Pred_ci,indx = zip(*sortedz)

short_array = []
for i in range(dspace):
    short_array.append(dPred[indx[i]])


mcci_in(short_array[0][0],short_array[0][1])
fout = open("data1.dat","w")
fout.write(str(dspace-1)+" "+str(nelec)+'\n')
for i in range(1,dspace):
    fout.write(mcci_format(short_array[i][0],short_array[i][1]))
fout.close()
fmcci.close()
subprocess.run("/home/madhumita/madhumita/molANN/mcci-master/stretchd_h2o/MLCI/varE/./varE",shell = True)



conf_E = np.loadtxt("csf_energy",usecols=(0))
ecore = np.loadtxt("ecore_value",usecols=(0))
min_value = min(conf_E)
min_index = np.argmin(conf_E)   # according to data1.dat order
en_tot = min_value + ecore

f = open("e_summary")
lines = f.readlines()
e_subspace = round(float(lines[71]),4)
f.close()
if (e_subspace > en_tot):
   print("lower energy")
   print(e_subspace,en_tot)
   
   fmcci = open("mcci.in","r+")
   mcci_in(short_array[min_index][0],short_array[min_index][1])
   poped_array = []
   for i in range(dspace):
       if i != min_index:
            poped_array.append(short_array[i])
   fout = open("data1.dat","w")
   len_pop = len(poped_array)
   fout.write(str(len_pop)+" "+str(nelec)+'\n')
   for i in range(len_pop):
       fout.write(mcci_format(poped_array[i][0],poped_array[i][1]))
   fout.close()
   subprocess.run("/home/madhumita/madhumita/molANN/mcci-master/stretchd_h2o/github/varE/./varE")
   fmcci.close()
# ----------------------------
