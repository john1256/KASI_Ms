from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad as quad_vec
from tqdm import tqdm
import os
import pandas as pd
Field_research_path = os.path.dirname(os.path.dirname(__file__))
# load dataset
sndata = pd.read_csv(Field_research_path + '/Data/parsonage.txt', sep=' ', engine='python')
z = sndata['zcmb'].values
mb = sndata['mb'].values
dmb = sndata['dmb'].values
grid = np.linspace(z.min(), z.max(), 100)

def E_inverse_flat(z, Omega_m): # return 1/E(z) = H0/H(z)
    Omega_L = 1 - Omega_m
    E2 = Omega_m*(1+z)**3 + Omega_L
    E = np.sqrt(E2)
    return 1/E


def Other_stuff_flat(parm): # parm[0] = Omegam, parm[1] = Omegalamb
    """
    Other_stuff_flat(z, parm) = integral from 0 to z of 1/E(z) dz
    Other_stuff_flat(z, parm) = H0/c*D_M(z)"""
    Omegam = parm[0]
    grid_dist = np.array([quad_vec(E_inverse_flat, 0,n, args=(Omegam))[0] for n in grid])
    interp_func = interp1d(grid, grid_dist, kind='linear', fill_value='extrapolate')
    integral = interp_func(z)
    return integral
def B(func, parm):
    """
    B(Omegam, Omegalamb) = 5*log10((1+z)*proper distance*H0/c)
    m(z) = A + B(Omegam, Omegalamb)
    input : 
        func : proper distance*H0/c (Other_stuff_flat or Other_stuff_curved)
        parm : [Omegam, Omegalamb] 
        z : redshift
    output :
        Bval : B(Omegam, Omegalamb)
    """
    funcval = func(parm) # proper distance*H0/c
    Bval = 5*np.log10((1+z)*funcval)
    return Bval

def A(mb, dmb,Bval):
    inv_dmb = np.sum(1/dmb**2)
    A = 1/inv_dmb*np.sum((mb - Bval)/(dmb**2))
    return A
# 1. make a code that accounts for a prior
def ln_prior(min,max):
    volume = np.prod(np.abs(min - max)) # volume of the prior
    return np.log(1/volume)

def Loglikelihood(func, parm): # return Loglikelihood = -chisq, parm[0] = H0, parm[1] = Omegam, parm[2] = Omegalamb
    Bval = B(func, parm)
    if np.isnan(Bval).any():
        print("Bval is NaN, returning -inf")
        print(f"parm = {parm}")
        return -np.inf
    m_z = A(mb,dmb,Bval) + Bval # m_z = A + B(Omegam, Omegalamb)
    diff = (mb - m_z)**2
    chisq = np.sum(diff/dmb**2)
    return -chisq/2

def ln_f(func, parm, prior, lnprior): # return total Loglikelihood
    bool = np.all((prior[0] <= parm) & (parm <= prior[1]))
    if bool == True:
        return lnprior + Loglikelihood(func, parm) # param[0] = H0, param[1] = Omegam, param[2] = Omegalamb
    else:
        return -np.inf

def Markov(func, paramk,paramkp1, prior, lnprior):
    minuschisqk = ln_f(func, paramk, prior, lnprior)
    minuschisqkp1 = ln_f(func, paramkp1, prior, lnprior)
    lnr = np.log(np.random.uniform(0.,1.))

    if minuschisqkp1 - minuschisqk > lnr:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : True")
        return paramkp1, minuschisqkp1
    else:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : False")
        return paramk, minuschisqk

def MCMC(func, paraminit,nstep,normal_vec,prior): # param0 = [H0, Omegam, Omegalamb]
    lnprior = ln_prior(prior[0], prior[1]) # calculate the prior volume likelihood
    param0 = paraminit
    arr = np.zeros((len(param0) + 1,nstep))
    stepsize = normal_vec
    for k in tqdm(range(nstep)):
        paramkp1 = np.array(param0 + np.random.normal(0,stepsize))
        param0, loglikelihood = Markov(func, param0, paramkp1, prior, lnprior) #loglikelihood = -chisq
        col = np.hstack((param0, loglikelihood))
        arr[:,k] = col
    return arr