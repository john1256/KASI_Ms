import numpy as np
import cosmoprimo
import pandas as pd
from tqdm import tqdm

dr1_data = pd.read_csv('BAOdata/dr1_mean.txt', comment='#', delim_whitespace=True, header=None)
dr1_cov = np.loadtxt('BAOdata/dr1_cov.txt')
dr2_data = pd.read_csv('BAOdata/dr2_mean.txt', comment='#', delim_whitespace=True, header=None)
dr2_cov = np.loadtxt('BAOdata/dr2_cov.txt')

def hmc_sample(U, grad_U, epsilon, L, current_q, std_dev):
    '''
         U: function returns the potential energy given a state q
         grad_u: function returns gradient of U given q
         epsilon: step size
         L: number of Leapfrog steps
         current_q: current generalized state trajectory starts from
         std_dev: vector of standard deviations for Gaussian (hyperparameter)
    '''
    q = current_q
    p = np.random.normal(np.zeros(len(q)), std_dev)  # sample zero-mean Gaussian
    current_p = p

    # Leapfrog: half step for momentum
    p = p - epsilon * grad_U(q) / 2

    for i in range(0, L):
        # Leapfrog: full step for position
        q = q + epsilon * p

        # Leapfrog: combine 2 half-steps for momentum across iterations
        if (i != L-1):
            p = p - epsilon * grad_U(q)

    # Leapfrog: final half step for momentum
    p = p - epsilon * grad_U(q)

    # Negate trajectory to make proposal symmetric (a no-op)
    p = -p

    # Compute potential and kinetic energies
    current_U = U(current_q)
    current_K = np.sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = np.sum(p ** 2) / 2
    
    # Accept with probability specified using Equation 44:
    if np.random.rand() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q, proposed_U,1
    else:
        return current_q, current_U,0

from cosmoprimo import Cosmology
from scipy.stats import norm

cosmo = Cosmology(engine='class') # LCDM
#cosmo = cosmo.clone(w0_fld=-0.42, wa_fld=-1.75) # w0waCDM

def chisq(q):
    theta = q.copy()
    hr_d = theta.pop('hr_d')
    cosmo_vary = cosmo.clone(**theta)
    speed_of_light = 2.99792458e5  # in km/s
    ba = cosmo_vary.get_background()
    
    # Model prediction
    def get_model(values, data):
        ind = data[2].values == values
        redshifts = data[0].values[ind]
        if values == 'DV_over_rs':
            model = (ba.comoving_angular_distance(redshifts) ** 2 / cosmo_vary.get('h') ** 2\
                      *  speed_of_light * redshifts / ba.hubble_function(redshifts))**(1/3) *cosmo_vary.get('h')/ hr_d
        elif values == 'DM_over_rs':
            model = ba.comoving_angular_distance(redshifts) / hr_d
        elif values == 'DH_over_rs':
            model = speed_of_light * cosmo_vary.get('h') / (ba.hubble_function(redshifts) * hr_d)
        return model, ind
    def set_model_vector(data):
        model_vec = np.zeros(len(data))
        dv_model, dv_zind = get_model('DV_over_rs', data)
        dm_model, dm_zind = get_model('DM_over_rs', data)
        dh_model, dh_zind = get_model('DH_over_rs', data)
        model_vec[dv_zind] = dv_model
        model_vec[dm_zind] = dm_model
        model_vec[dh_zind] = dh_model
        return model_vec
    #model_dr1 = set_model_vector(dr1_data)
    model_dr2 = set_model_vector(dr2_data)
    # Chi-squared calculation
    #delta_dr1 = dr1_data[1].values - model_dr1
    delta_dr2 = dr2_data[1].values - model_dr2
    #chi2_dr1 = delta_dr1.T @ np.linalg.inv(dr1_cov) @ delta_dr1
    chi2_dr2 = delta_dr2.T @ np.linalg.inv(dr2_cov) @ delta_dr2
    return chi2_dr2 #+ chi2_dr1
def prior(theta):
    for key, value in theta.items():
        if key == 'Omega_cdm' and not (0.1 < value < 0.5):
            return 0
        if key == 'Omega_b' and not (0.01 < value < 0.1):
            return 0
        if key == 'h' and not (0.5 < value < 0.9):
            return 0
        return 1
def U(theta): 
    if prior(theta) == 0:
        return -np.inf
    chi2 = chisq(theta)
    #print(0.5 * (chi2 + prior_val))
    return 0.5 * (chi2)

def gradU(theta, h=1e-5):
    grad = {}
    base_U = U(theta)
    for key in theta.keys():
        theta_h = theta.copy()
        theta_h[key] = theta[key] + h
        
        #print(theta_h[key] - theta[key])
        U_h = U(theta_h)     
        grad[key] = (U_h - base_U) / h
    #print(grad)
    return pd.Series(grad)





# HMC sampling 1 
# epsilon = 3e-3
# L = 30; N = 1000
# std_dev = [1.,1.]

# HMC sampling 2
# epsilon = 1e-3
# L=30; N = 1000
# std_dev = [1.,1.]
"""
sample = []
accepts = 0
epsilon = 1e-3
L = 30; N = 1000
current_q = pd.Series({'Omega_m' : 0.296, 'hr_d' : 101.48})  + pd.Series({'Omega_m' : np.random.normal(0,0.01), 'hr_d' : np.random.normal(0,0.1)})
std_dev = [1.,1.]

samples= []; likelihoods=[]
accepts = 0
for i in tqdm(range(0, N,1), desc="HMC Sampling 2"): 
    sample, like,accept = hmc_sample(U, gradU, epsilon, L, current_q, std_dev)
    samples.append(sample); likelihoods.append(like)
    accepts += accept
    current_q = samples[-1]

samples = pd.DataFrame(samples)
samples.to_csv('mcmc/HMC_DESI_samples_2.csv', index=False)
likelihoods = np.array(likelihoods)
np.save('mcmc/HMC_DESI_likelihoods_2', likelihoods)
np.savetxt('mcmc/HMC_DESI_acceptance_num_2.txt', np.array([accepts]))
# HMC sampling 3
# epsilon = 3e-3
# L=30; N = 1000
# std_dev = [2.,2.]
sample = []
accepts = 0
epsilon = 3e-3
L = 30; N = 1000
current_q = pd.Series({'Omega_m' : 0.296, 'hr_d' : 101.48})  + pd.Series({'Omega_m' : np.random.normal(0,0.01), 'hr_d' : np.random.normal(0,0.1)})
std_dev = [2.,2.]

samples= []; likelihoods=[]
accepts = 0
for i in tqdm(range(0, N,1), desc="HMC Sampling 3"): 
    sample, like,accept = hmc_sample(U, gradU, epsilon, L, current_q, std_dev)
    samples.append(sample); likelihoods.append(like)
    accepts += accept
    current_q = samples[-1]

samples = pd.DataFrame(samples)
samples.to_csv('mcmc/HMC_DESI_samples_3.csv', index=False)
likelihoods = np.array(likelihoods)
np.save('mcmc/HMC_DESI_likelihoods_3', likelihoods)
np.savetxt('mcmc/HMC_DESI_acceptance_num_3.txt', np.array([accepts]))
# HMC sampling 4
# epsilon = 3e-3
# L=30; N = 7000
# std_dev = [1.,1.]
sample = []
accepts = 0
epsilon = 3e-3
L = 30; N = 7000
current_q = pd.Series({'Omega_m' : 0.296, 'hr_d' : 101.48})  + pd.Series({'Omega_m' : np.random.normal(0,0.01), 'hr_d' : np.random.normal(0,0.1)})
std_dev = [1.,1.]

samples= []; likelihoods=[]
accepts = 0
for i in tqdm(range(0, N,1), desc="HMC Sampling 4"): 
    sample, like,accept = hmc_sample(U, gradU, epsilon, L, current_q, std_dev)
    samples.append(sample); likelihoods.append(like)
    accepts += accept
    current_q = samples[-1]

samples = pd.DataFrame(samples)
samples.to_csv('mcmc/HMC_DESI_samples_4.csv', index=False)
likelihoods = np.array(likelihoods)
np.save('mcmc/HMC_DESI_likelihoods_4', likelihoods)
np.savetxt('mcmc/HMC_DESI_acceptance_num_4.txt', np.array([accepts]))
"""
# HMC sampling 5
# epsilon = 1e-3
# L=30; N=1000
# std_dev = [2.,2.]
sample = []
accepts = 0
epsilon = 1e-3
L = 30; N = 1000
current_q = pd.Series({'Omega_m' : 0.296, 'hr_d' : 101.48})  + pd.Series({'Omega_m' : np.random.normal(0,0.01), 'hr_d' : np.random.normal(0,0.1)})
std_dev = [2.,2.]

samples= []; likelihoods=[]
accepts = 0
for i in tqdm(range(0, N,1), desc="HMC Sampling 5"): 
    sample, like,accept = hmc_sample(U, gradU, epsilon, L, current_q, std_dev)
    samples.append(sample); likelihoods.append(like)
    accepts += accept
    current_q = samples[-1]

samples = pd.DataFrame(samples)
samples.to_csv('mcmc/HMC_DESI_samples_5.csv', index=False)
likelihoods = np.array(likelihoods)
np.save('mcmc/HMC_DESI_likelihoods_5', likelihoods)
np.savetxt('mcmc/HMC_DESI_acceptance_num_5.txt', np.array([accepts]))