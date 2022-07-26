#!/usr/bin/python3
import numpy as np
from pandas import read_csv

def sensi_speci(atype,cutoff):
    fout = open("file","a")
    if atype == "validation_space":
        det_num = np.loadtxt("validation.out",usecols=0,dtype=int)
        ci_pred = np.loadtxt("validation.out",usecols=4)
        ci = np.loadtxt("validation.out",usecols=3)
        length=len(ci)
        det_ref = det_num

    elif atype == "whole_space":
        det_num = np.loadtxt("Pred.out",usecols=(0),dtype=int)
        ci_pred = np.loadtxt("Pred.out",usecols=3)
        ci  = read_csv("csv_data/whole_space.csv",usecols=[47], header=None).values[:,-1]
        length=len(ci)
        det_ref = range(1,length+1)


    positive_pred=[]
    negetive_pred=[]
    positive=[]
    negetive=[]

    length=len(ci)
    for i in range(length):
        if (abs(ci_pred[i]) < cutoff):
            negetive_pred.append(det_num[i])
        if (abs(ci_pred[i]) >= cutoff):
            positive_pred.append(det_num[i])
        if (abs(ci[i]) < cutoff):
            negetive.append(det_ref[i])
        if (abs(ci[i]) >= cutoff):
            positive.append(det_ref[i])

    true_positive = len(set(positive) & set(positive_pred))
    true_negetive = len(set(negetive) & set(negetive_pred))
    false_positive = len(positive_pred) - true_positive
    false_negetive = len(negetive_pred) - true_negetive

    sensi = (true_positive*1.0)/(true_positive+false_negetive)
    spesi = (true_negetive*1.0)/(true_negetive+false_positive)
    sensi = sensi*100
    spesi = spesi*100

    fout.write('\n'+str(cutoff)+"    "+str(sensi)+"      "+str(spesi)+'\n')
    fout.write("true_positive  : %7d, true_negetive  : %7d" % (true_positive, true_negetive)+'\n')
    fout.write("false_positive : %7d, false_negetive : %7d" % (false_positive, false_negetive)+'\n')
    fout.close()


sensi_speci("whole_space",0.001)    # whole_space/validation_space
