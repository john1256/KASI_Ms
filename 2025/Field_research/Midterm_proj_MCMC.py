import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import multiprocess as mp
sys.path.append('./2025/Field_research/Utils')

import utils_flat as uf
import utils_flat_SNBAO as ufb
import utils_curved_SNBAO as ucb
import utils_curved as uc




CurrentPath = os.getcwd()
sndata = pd.read_csv(CurrentPath + '/2025/Field_research/Data/parsonage.txt', sep = ' ', engine='python')
# BAO data
BAO_z = np.array([0.094,0.157,0.402,0.402,0.526,0.526,0.597,0.597])
BAO_val = np.array([0.08,1849.05,4006.83,43.48,4650.20,51.31,5053.80,58.64])
BAO_err = np.array([0.003,66.648,59.924,1.334,68.103,1.512,75.657,1.292])
BAO_ind = np.array([1,2,3,4,3,4,3,4]) 
# 1 : r_s/D_V, 2 : D_V(r_d,fid/r_d), 3 : D_M(r_d,fid/r_d), 4 : H(r_d/r_d,fid)
BAO_data = pd.DataFrame({
    'z': BAO_z,
    'val': BAO_val,
    'err': BAO_err,
    'ind': BAO_ind
})

# SN data

# flat
def run_mcmc_flat_SN(seed):
    np.random.seed(seed)
    Prior1 = np.array([[0.],[1.]])
    normal_vec = np.array([0.0003])*10
    std = np.array([0])*0.1
    paraminit = np.array([0.9999742233029345]) + np.random.normal(0,std)
    nstep = int(1e6)
    return uf.MCMC(uf.Other_stuff_flat, paraminit, sndata, nstep, normal_vec, Prior1)

# curved
def run_mcmc_curved_SN(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., -3.],[20, 1.]])
    std = np.array([0.7765663, 0.5880332])
    normal_vec = np.array([1., 0.2])
    paraminit = np.array([3.1, 0. ]) + np.random.normal(0,std)
    nstep = int(2*1e5)
    return uc.MCMC(uc.Other_stuff_curved, paraminit, sndata, nstep, normal_vec, Prior2)

# SN + BAO data

# flat
def run_mcmc_flat_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., 4.66594018],[1, 36.39070654]])
    std = np.array([0.,  0.11162995])
    normal_vec = np.array([0.0003614, 0.11162995])*2.
    paraminit = np.array([0.9999742233029345,  31.689002893196257]) + np.random.normal(0,std)
    nstep = int(1.5*1e5)
    return ufb.MCMC_BAO(ufb.Other_stuff_flat, ufb.BAO_flat, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)

# curved
def run_mcmc_curved_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., -3., 4.66594018],[20, 1., 35.39070654]])
    std = np.array([0.2096627, 0.0, 2.945705])
    normal_vec = np.array([0.7, 0.1, 1])*0.5
    paraminit = np.array([3.5,  0.1,20]) + np.random.normal(0,std)
    nstep = int(1e5)
    return ucb.MCMC_BAO(ucb.Other_stuff_curved, ucb.BAO_curved, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)


# execution    
n_chain = 4
# SN flat
"""
print("Running MCMC for SN flat...")
with mp.Pool(processes=n_chain) as pool:
    results1 = pool.map(run_mcmc_flat_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/Field_research/Results/MCMC_flat_SN_{i}.npy', results1[i])
"""

# SN curved
print("Running MCMC for SN curved...")
with mp.Pool(processes=n_chain) as pool:
    results2 = pool.map(run_mcmc_curved_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/Field_research/Results/MCMC_curved_SN_{i}.npy', results2[i])

"""
# SN + BAO flat
print("Running MCMC for SN + BAO flat...")
with mp.Pool(processes=n_chain) as pool:
    results3 = pool.map(run_mcmc_flat_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/Field_research/Results/MCMC_flat_SN+BAO_{i}.npy', results3[i])
"""
"""
# SN + BAO curved
print("Running MCMC for SN + BAO curved...")
with mp.Pool(processes=n_chain) as pool:
    results4 = pool.map(run_mcmc_curved_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/Field_research/Results/MCMC_curved_SN+BAO_{i}.npy', results4[i])
"""
