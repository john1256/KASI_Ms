from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import quad_vec
from tqdm import tqdm
import os
import pandas as pd
Field_research_path = os.path.dirname(os.path.dirname(__file__))
# load sn dataset
sndata = pd.read_csv(Field_research_path + '/Data/parsonage.txt', sep=' ', engine='python')
z = sndata['zcmb'].values
mb = sndata['mb'].values
dmb = sndata['dmb'].values

# load BAO dataset
BAO_data = pd.read_csv(Field_research_path + '/Data/BAO_data.csv')
z_BAO = BAO_data['z'].values
ind_BAO = BAO_data['ind'].values
y0_BAO = BAO_data['val'].values
err_BAO = BAO_data['err'].values
# SN analysis for flat universe
grid = np.linspace(z.min(), z.max(), 100)

def E_inverse_curved(z, Omega_m, Omega_L): # return 1/E(z) = H0/H(z)
    Omegak = 1 - Omega_m - Omega_L
    E2 = Omega_m*(1+z)**3 + Omega_L + Omegak*(1+z)**2
    E = np.sqrt(E2) 
    return 1/E


def Other_stuff_curved(parm): # parm[0] = Omegam, parm[1] = Omegalamb, d_p = c/H0*Other_stuff_flat
    Omegam = parm[0]
    Omega_L = parm[1]
    Omegak = 1 - Omegam - Omega_L
    gridval = np.array([quad_vec(E_inverse_curved, 0,n, args=(Omegam, Omega_L))[0] for n in grid])
    if np.abs(Omegak) < 1e-14: # flat universe
        grid_dist = gridval
    elif Omegak > 1e-14: # open universe
        grid_dist = 1/np.sqrt(Omegak)*np.sinh(np.sqrt(Omegak)*gridval)
    else: # closed universe
        grid_dist = 1/np.sqrt(-Omegak)*np.sin(np.sqrt(-Omegak)*gridval)
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
#    ndata = mb.size
    inv_dmb = np.sum(1/dmb**2)
    A = 1/inv_dmb*np.sum((mb - Bval)/(dmb**2))
    return A

# BAO analysis for flat universe
def D_M_curved(z,parm): # return D_M(z) = c/H0*Other_stuff_flat
    Omegam, Omegalamb, H0 = parm
    Omegak = 1 - Omegam - Omegalamb
    c = 299792.458 # speed of light in km/s
    integral = np.array([quad_vec(E_inverse_curved, 0,zval, args=(Omegam,Omegalamb))[0] for zval in z])
    if np.abs(Omegak) < 1e-14: # flat universe
        D_M = integral*c/H0
    elif Omegak > 1e-14: # open universe
        D_M = 1/np.sqrt(Omegak)*np.sinh(np.sqrt(Omegak)*integral)*c/H0
    else: # closed universe
        D_M = 1/np.sqrt(-Omegak)*np.sin(np.sqrt(-Omegak)*integral)*c/H0
    return D_M
def D_V_curved(z, parm):
    c = 299792.458 # speed of light in km/s
    Omegam, Omegalamb, H0 = parm
    D_M = D_M_curved(z, parm)
    E_inv = np.array([E_inverse_curved(zval, Omegam, Omegalamb) for zval in z])
    D_V = (c*z/H0*(D_M**2)*E_inv)**(1/3)
    return D_V
def z_eq(parm):
    Omegam, _, H0 = parm
    h = H0/100 # dimensionless Hubble constant
    return Omegam*h**2/(1.48*10**-6)
def R_eq():
    f = 0.1543  # baryonic fraction
    R_eq = 3/4*f
    return R_eq
def R_rec(z_eq):
    f = 0.1543  # baryonic fraction
    z_rec = 2426.3839
    R_rec = 3/4*f*(1+z_eq)/(1+z_rec)
    return R_rec

def r_dfid(parm):
    """
    r_dfid = r_d(z_eq) = sound horizon at the drag epoch
    input : parm = [Omegam, H0]
    output : r_dfid
    """
    c = 299792.458 # speed of light in km/s
    Omegam, _,H0 = parm
    z_eq_val = z_eq(parm)
    R_eq_val = R_eq()
    R_rec_val = R_rec(z_eq_val)
    
    r_dfid_val = (
    1/np.sqrt(Omegam*H0**2) * 2*c / np.sqrt(3*z_eq_val*R_eq_val)* 
    np.log((np.sqrt(1+R_rec_val) + np.sqrt(R_rec_val + R_eq_val))/(1+ np.sqrt(R_eq_val)))
    )
    return r_dfid_val

def BAO_curved(parm): # return y for BAO data
    r_dfid_val = r_dfid(parm)
    r_d = 103.16 # Mpc
    result = np.zeros(z_BAO.size)
    # type 1 : r_d/D_V
    ind1 = np.where(ind_BAO == 1)
    z1 = z_BAO[ind1]
    D_V_val = D_V_curved(z1, parm)
    result[ind1] = r_d/D_V_val
    # type 2 : D_V(r_dfid/r_d)
    ind2 = np.where(ind_BAO == 2)
    z2 = z_BAO[ind2]
    D_V_val = D_V_curved(z2, parm)
    result[ind2] = D_V_val*(r_dfid_val/r_d)
    # type 3 : D_M(r_dfid/r_d)
    ind3 = np.where(ind_BAO == 3)
    z3 = z_BAO[ind3]
    D_M_val = D_M_curved(z3, parm)
    result[ind3] = D_M_val*(r_dfid_val/r_d)
    # type 4 : H(r_d/r_dfid)
    ind4 = np.where(ind_BAO == 4)
    z4 = z_BAO[ind4]
    E_inverse_arr= np.array([E_inverse_curved(zval, parm[0], parm[1]) for zval in z4])
    H_val = 1/E_inverse_arr * parm[2] # 1/E(z) = H0/H(z) -> H(z) = H0*E(z)
    result[ind4] = H_val*(r_d/r_dfid_val)
    return result

# MCMC analysis
def ln_prior(min,max):
    volume = np.prod(np.abs(min - max)) # volume of the prior
    return np.log(1/volume)

def SN_Loglikelihood(func, parm): # return Loglikelihood = -chisq, parm[0] = H0, parm[1] = Omegam, parm[2] = Omegalamb
    Bval = B(func, parm) # B(Omegam, Omegalamb)
    if np.isnan(Bval).any():
        print("Bval is NaN, returning -inf")
        print(f"parm = {parm}")
        return -np.inf
    m_z = A(mb,dmb,Bval) + Bval # m_z = A + B(Omegam, Omegalamb)
    diff = (mb - m_z)**2
    chisq = np.sum(diff/dmb**2)
    return -chisq/2
def BAO_loglikelihood(func, parm): # return Loglikelihood = -chisq

    y_BAO = func(parm)
    diff = (y_BAO - y0_BAO)**2
    chisq = np.sum(diff/err_BAO**2)
    return -chisq/2

def ln_f(func_SN,func_BAO, parm, prior, lnprior): # return total Loglikelihood
    bool = np.all((prior[0] <= parm) & (parm <= prior[1])) # Prior : [[Omegam, H0],[Omegam, H0]]
    if bool == True:
        return lnprior + SN_Loglikelihood(func_SN, parm[:2]) + BAO_loglikelihood(func_BAO, parm) # param[0] = Omegam, param[1] = H0 (Marginalized parameter)
    else:
        return -np.inf

def Markov_SNBAO(func_SN, func_BAO, paramk,paramkp1, prior, lnprior):
    minuschisqk = ln_f(func_SN, func_BAO, paramk, prior, lnprior)
    minuschisqkp1 = ln_f(func_SN, func_BAO, paramkp1, prior, lnprior)
    lnr = np.log(np.random.uniform(0.,1.))

    if minuschisqkp1 - minuschisqk > lnr:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : True")
        return paramkp1, minuschisqkp1
    else:
#        print(f"param0 = {paramk}, paramkp1 = {paramkp1}, \n chisq0 = {minuschisqk}, chisqkp1 = {minuschisqkp1}, lnr = {lnr}, moved : False")
        return paramk, minuschisqk

def MCMC_SNBAO(func_SN, func_BAO, paraminit, nstep,normal_vec,prior): # param0 = [H0, Omegam, Omegalamb]
    """_summary_

    Args:
        func_SN (function): Other_stuff_flat
        func_BAO (function): BAO_flat
        paraminit (array): _description_
        SNdata (DataFrame): _description_
        BAOdata (DataFrame): _description_
        nstep (int): _description_
        normal_vec (array): _description_
        prior (array): _description_

    Returns:
        arr: mcmc arr before marginalization of H0
    """
    lnprior = ln_prior(prior[0], prior[1]) # calculate the prior volume likelihood
    param0 = paraminit
    arr = np.zeros((len(param0) + 1,nstep))
    stepsize = normal_vec
    for k in tqdm(range(nstep)):
        paramkp1 = np.array(param0 + np.random.normal(0,stepsize))
        param0, loglikelihood = Markov_SNBAO(func_SN, func_BAO, param0, paramkp1, prior, lnprior) #loglikelihood = -chisq
        col = np.hstack((param0, loglikelihood))
        arr[:,k] = col
    return arr