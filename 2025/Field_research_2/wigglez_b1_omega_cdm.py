import os
import numpy as np
import cosmoprimo

# data dir in NERSC

NERSC_dir = '/pscratch/sd/j/john0712'
wigglez_data_dir = os.path.join(NERSC_dir, 'Practices','wigglez')
z_effs = [0.22, 0.41, 0.60, 0.78]

def read_wigglez_files(file_type,session, region='9-hr', data_dir = wigglez_data_dir):
    """a : 0.1<z<0.3, b : 0.3<z<0.5, c : 0.5<z<0.7, d : 0.7<z<0.9"""
    if session not in ['a','b', 'c', 'd']:
        
        raise ValueError("Session must be one of 'a', 'b', 'c', or 'd'")
    elif file_type not in ['cov', 'measurements', 'windows']:
        raise ValueError("File type must be one of 'cov', 'measurements', or 'windows'")
    filepath = os.path.join(data_dir, f'wigglez_jan11{session}_{file_type}.txt')
    with open(filepath, 'r') as f:
        data_lines = []
        prev_line_is_header = False
        for line in f:
            if line.startswith('#') and not prev_line_is_header:
                if 'region,' in line:
                    headers = line.split()
                    region_name = headers[headers.index('region,') - 1]
                    if region_name == region:
                        prev_line_is_header = True
                else:
                    continue
            elif line.startswith('#') and prev_line_is_header:
                if 'k / hMpc^-1' in line:
                    continue
                else:
                    break
            else:
                if prev_line_is_header:
                    line_arr = line.split()
                    line_arr = [float(x) for x in line_arr]
                    data_lines.append(line_arr)
                else:
                    continue
    return np.array(data_lines)

kbin=(0,100)

def get_wigglez_dataset(dataset = 'a'):
    klim = (2,20); 
    if dataset == 'a':
        regions = ['9-hr']
    else:
        regions = ['9-hr', '11-hr', '15-hr', '22-hr', '0-hr', '1-hr', '3-hr']

    cov = [read_wigglez_files('cov',dataset, region=reg)[klim[0]:klim[1],klim[0]:klim[1]] for reg in regions]
    
    k_measured = [read_wigglez_files('measurements',dataset, region=reg)[klim[0]:klim[1],0] for reg in regions]
    power_measured = [read_wigglez_files('measurements',dataset, region=reg)[klim[0]:klim[1],3] for reg in regions]
    windows = [read_wigglez_files('windows',dataset, region=reg)[klim[0]:klim[1], kbin[0]:kbin[1]] for reg in regions]
    return k_measured, power_measured, windows, cov

kbands = np.loadtxt(os.path.join(wigglez_data_dir, 'wigglez_jan11kbands.txt'))[kbin[0]:kbin[1]]

from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from scipy.linalg import block_diag
from desilike import setup_logging
from desilike.theories import Cosmoprimo

def make_wigglez_likelihood(dataset='a', varied_params=None):
    k_measured, power_measured, windows, cov = get_wigglez_dataset(dataset)
    wigglez_cosmo = cosmoprimo.Cosmology(
        omega_cdm=0.1153,
        omega_b=0.02238,
        h=0.689,
        tau_reio = 0.083,
        n_s = 0.964,
        logA = 3.084, engine='class'
    )
    #wigglez_cosmo = Cosmoprimo(fiducial='DESI', **wigglez_params)
    data_name = ['a', 'b', 'c', 'd']
    k_measured, power_measured, windows, cov = get_wigglez_dataset(dataset)
    template = DirectPowerSpectrumTemplate(fiducial=wigglez_cosmo, z=z_effs[data_name.index(dataset)])
    #for param in list(set(['omega_cdm', 'omega_b', 'h', 'tau_reio', 'n_s', 'logA']) - set(varied_params)):
    #    template.params[param].update(fixed=True)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template, k=kbands)
    observables = []
    cov_tot = block_diag(*cov)
    for i in range(len(k_measured)):
        observable = TracerPowerSpectrumMultipolesObservable(theory=theory, data=power_measured[i], wmatrix=windows[i], ells=(0,), k=k_measured[i], kin=kbands, ellsin=(0,))
        observables.append(observable)
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov_tot)
    for param in list(set(['omega_cdm', 'omega_b', 'h', 'tau_reio', 'n_s', 'logA', 'b1', 'sn0']) - set(varied_params)):
        likelihood.all_params[param].update(fixed=True)
    return likelihood
    


setup_logging()
for dataset in ['c', 'd']:
    likelihood = make_wigglez_likelihood(dataset, varied_params=['omega_cdm', 'b1'])
    likelihood.all_params['b1'].update(prior={'limits': [0., 5.], 'latex': 'b_1'})
    likelihood.all_params['omega_cdm'].update(prior={'limits': [0.01, 0.4], 'latex': r'\omega_{cdm}'})
    #likelihood.all_params['h'].update(prior={'limits': [0.4, 1.0], 'latex': 'h'})
    
    from desilike.samplers import MCMCSampler
    sampler = MCMCSampler(likelihood,save_fn=f'_tests/chain_wigglez_{dataset}_b1_omega_cdm.npy')
    sampler.run(check={'max_eigen_gr': 0.2}, max_iterations=10000)