import numpy as np
import matplotlib.pyplot as plt

#compute autocorrelation table up to M steps
def autocorr_table(M,sample):
    N=len(sample)
    k = np.linspace(0,M,M,dtype=int)
    CC = np.zeros_like(k,dtype=np.float64)
    for kk in k:
        factor1 = sample[:N-kk]*sample[kk:]
        factor2 = sample[kk:].mean() * sample[:N-kk].mean()
        out=np.sum(factor1 - factor2)/(N-kk)
        CC[kk-1] = out

    return CC

def tau(sample):
    #compute autocorrelation table for whole sample
    table = autocorr_table(len(sample)-1,sample)

    #define array of taus and fill it buy summing cumulatively (like integrating autocorrelation function)
    tau = np.zeros_like(table)
    tau[0]=table[0]
    for i in range(0,len(tau)-1):
        tau[i+1] = tau[i] + table[i+1]

    tau = np.asarray(tau)

    return tau