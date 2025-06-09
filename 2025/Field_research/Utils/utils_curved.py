from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad as quad_vec
from tqdm import tqdm


def E_inverse_curved(z, Omega_m, Omegalamb): # return 1/E(z) = H0/H(z)
    Omegak = 1 - Omega_m-Omegalamb
    E2 = Omega_m*(1+z)**3 + Omegalamb + Omegak*(1+z)**2
    if (E2 <0).any():
        return np.nan
    E = np.sqrt(E2)
    return 1/E


def Other_stuff_curved(z, parm): # parm[0] = Omegam, parm[1] = Omegalamb
    Omegam = parm[0]
    Omegalamb = parm[1]
    Omegak = 1 - Omegam - Omegalamb
    grid = np.linspace(z.min(), z.max(), 100)
    grid_Ez = np.array([quad_vec(E_inverse_curved, 0,n, args=(Omegam, Omegalamb))[0] for n in grid])
    if np.abs(Omegak) < 1e-14: # flat universe
        grid_dist = grid_Ez
    elif Omegak > 1e-14: # open universe
        grid_dist = 1/np.sqrt(Omegak)*np.sinh(np.sqrt(Omegak)*grid_Ez)
    else: # closed universe
        grid_dist = 1/np.sqrt(-Omegak)*np.sin(np.sqrt(-Omegak)*grid_Ez)
    interp_func = interp1d(grid, grid_dist, kind='linear', fill_value='extrapolate')
    integral = interp_func(z)
    return integral
def B(func, parm,z):
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
    funcval = func(z, parm) # proper distance*H0/c
    if (funcval).any() <= 0 or np.isnan(funcval).any():
        funcval= np.nan
    Bval = 5*np.log10((1+z)*funcval)
    return Bval

def A(func,mb, dmb,z, parm):
#    ndata = mb.size
    inv_dmb = np.sum(1/dmb**2)
    A = 1/inv_dmb*np.sum((mb - B(func,parm,z))/(dmb**2))
    return A

# 1. make a code that accounts for a prior
def ln_prior(min,max):
    volume = np.prod(np.abs(min - max)) # volume of the prior
    return np.log(1/volume)

def Loglikelihood(func, parm,SNdata): # return Loglikelihood = -chisq, parm[0] = H0, parm[1] = Omegam, parm[2] = Omegalamb
    mb = SNdata['mb'].values
    dmb = SNdata['dmb'].values
    z = SNdata['zcmb'].values
    m_z = A(func, mb,dmb, z, parm) + B(func, parm, z) # m_z = A + B(Omegam, Omegalamb)
    diff = (mb - m_z)**2
    chisq = np.sum(diff/dmb**2)
    return -chisq/2

def ln_f(func, parm,SNdata, prior, lnprior): # return total Loglikelihood
    bool = np.all((prior[0] <= parm) & (parm <= prior[1]))
    if bool == True:
        return lnprior + Loglikelihood(func, parm, SNdata) # param[0] = H0, param[1] = Omegam, param[2] = Omegalamb
    else:
        return -np.inf

def Markov(func, paramk,paramkp1,SNdata, prior, lnprior):
    minuschisqk = ln_f(func, paramk, SNdata, prior, lnprior)
    minuschisqkp1 = ln_f(func, paramkp1, SNdata, prior, lnprior)
    lnr = np.log(np.random.uniform(0.,1.))

    if minuschisqkp1 - minuschisqk > lnr:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : True")
        return paramkp1, minuschisqkp1
    else:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : False")
        return paramk, minuschisqk

def MCMC(func, paraminit,SNdata, nstep,normal_vec,prior): # param0 = [H0, Omegam, Omegalamb]
    lnprior = ln_prior(prior[0], prior[1]) # calculate the prior volume likelihood
    param0 = paraminit
    arr = np.zeros((len(param0) + 1,nstep))
    stepsize = normal_vec
    for k in tqdm(range(nstep)):
        paramkp1 = np.array(param0 + np.random.normal(0,stepsize))
        param0, loglikelihood = Markov(func, param0, paramkp1,SNdata, prior, lnprior) #loglikelihood = -chisq
        col = np.hstack((param0, loglikelihood))
        arr[:,k] = col
    return arr