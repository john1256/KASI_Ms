import numpy as np
import cosmoprimo
import pandas as pd
dr1_data = pd.read_csv('BAOdata/dr1_mean.txt', comment='#', delim_whitespace=True, header=None)
dr1_cov = np.loadtxt('BAOdata/dr1_cov.txt')
dr2_data = pd.read_csv('BAOdata/dr2_mean.txt', comment='#', delim_whitespace=True, header=None)
dr2_cov = np.loadtxt('BAOdata/dr2_cov.txt')
def mh_sampler(f, proposal, steps, initial=0.0):
    x_curr = initial
    while True:
        accept = 0
        for i in range(steps):
            x_next = proposal(x_curr)
            if min(1, np.exp(-f(x_next) + f(x_curr))) > np.random.uniform(0, 1):
                x_curr = x_next
                accept += 1
            loglike = f(x_curr)
        yield x_curr,loglike,accept / steps
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
    model_dr1 = set_model_vector(dr1_data)
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
    else:
        prior_val = -prior(theta)
    chi2 = chisq(theta)
    #print(0.5 * (chi2 + prior_val))
    return 0.5 * (chi2 + prior_val)

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

def proposal(x_curr):
    next_vec = np.random.normal(x_curr, [0.01,0.01])
    return pd.Series({'Omega_m': next_vec[0], 'hr_d': next_vec[1]})
from tqdm import tqdm
steps = 1
N = 1000
current_q = pd.Series({'Omega_m' : 0.296, 'hr_d' : 101.48}) + pd.Series({'Omega_m' : np.random.normal(0,0.1), 'hr_d' : np.random.normal(0,1.)})
samples = []; loglikes = []
accepts = 0
sampler = mh_sampler(U, proposal, steps, initial=current_q)
for i in tqdm(range(0, N, 1), desc='MH Sampling'):
    sample, loglike, accept = next(sampler)
    samples.append(sample)
    loglikes.append(loglike)
    accepts += accept
    q = samples[-1]

samples = pd.DataFrame(samples)
samples.to_csv('mcmc/MH_DESI_samples.csv', index=False)
loglikes = np.array(loglikes)
np.save('mcmc/MH_DESI_loglikes.npy', loglikes)
np.savetxt('mcmc/MH_DESI_acceptance_num.txt', np.array([accepts/N]))