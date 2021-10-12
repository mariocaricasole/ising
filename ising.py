import numpy as np
import matplotlib.pyplot as plt
from autocorr_time import autocorr_table, tau

def ising(N=1,L=1,beta=1,initial_state=None):
    #check if you're passing an initial state and define initial array
    if(initial_state==None):
        s_ij = np.full([N,N],-1)
        eps = -2
        m = -1
    else:
        s_ij,eps,m,_,_ = initial_state
        N = s_ij.shape[0]
        L = L/10

    #define boltzmann probability distribution and energy difference evaluation
    def boltz(dE,beta):
        return np.exp(-beta*dE)

    def dE(i,j,array):
        if(i+1>array.shape[0]-1):
            ii = 0
        else:
            ii = i+1

        if(j+1>array.shape[1]-1):
            jj = 0
        else:
            jj = j+1

        nearest_neigh = np.array([array[i-1,j],array[ii,j],array[i,j-1],array[i,jj]])
        return 2*array[i,j]*nearest_neigh.sum()

    #start simulation
    eps_arr, m_arr = [],[]
    for k in range(L):
        i,j = np.random.randint(0,N),np.random.randint(0,N)
        dE_k = dE(i,j,s_ij)

        if(dE_k<0):
            s_ij[i,j]*=-1
            eps += dE_k/N**2
            m += 2*s_ij[i,j]/N**2
        elif(np.random.random()<boltz(dE_k,beta)):
            s_ij[i,j]*=-1
            eps += dE_k/N**2
            m += 2*s_ij[i,j]/N**2

        eps_arr.append(eps)
        m_arr.append(m)

    #start analysis on datasets
    e_arr = np.asarray(eps_arr)
    m_arr = np.asarray(m_arr)

    #find autocorrelation time and choose only final portion of the dataset accordingly
    tau_e = np.max(tau(e_arr))
    tau_m = np.max(tau(m_arr))

    n_e = L*int(1 - 1/(2*tau_e))
    n_m = L*int(1- 1/(2*tau_e))

    #evaluate averages and std and specific heat, susceptibility
    e_avg = np.mean(e_arr[n_e:])
    m_avg = np.mean(m_arr[n_m:])

    e_std = np.std(e_arr[n_e:])
    m_std = np.std(m_arr[n_m:])

    print(f"e_avg = {e_avg} +- {e_std}")
    print(f"m_avg = {m_avg} +- {m_std}")

    cs = N**2 * np.var(e_arr[n_e:])
    chi = N**2 * np.var(m_arr[n_m:])

    return s_ij,eps,m, cs, chi