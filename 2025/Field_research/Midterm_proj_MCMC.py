import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import multiprocess as mp
Field_research_path = os.path.dirname(__file__)
sys.path.append(Field_research_path + '/Utils')

import utils_flat as uf
import utils_flat_BAO as ufb
import utils_curved_BAO as ucb
import utils_flat_SNBAO as ufsb
import utils_curved_SNBAO as ucsb
import utils_curved as uc




CurrentPath = os.getcwd()
sndata = pd.read_csv(Field_research_path+ '/Data/parsonage.txt', sep = ' ', engine='python')
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
    nstep = int(1e5)
    return uc.MCMC(uc.Other_stuff_curved, paraminit, sndata, nstep, normal_vec, Prior2)

# BAO data

# flat
def run_mcmc_flat_BAO(seed):
    np.random.seed(seed)
    Prior1 = np.array([[0., 4.66594018],[1, 36.39070654]])
    normal_vec = np.array([0.01, 0.1])
    std = np.array([0,0.1])
    paraminit = np.array([0.9999742233029345, 31.658736569698007]) + np.random.normal(0,std)
    nstep = int(1e5)
    return ufb.MCMC_BAO(ufb.BAO_flat, paraminit, BAO_data, nstep, normal_vec, Prior1)

# curved
def run_mcmc_curved_BAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., -3., 4.66594018],[20, 1., 35.39070654]])
    std = np.array([0.30967346, 0.37778227, 0.44020213])
    normal_vec = np.array([0.0590695, 0.08, 0.11])* 2
    paraminit = np.array([3.518803845192434,  -0.19403381170893674,20.071034556246392]) + np.random.normal(0,std)
    nstep = int(2e5)
    return ucb.MCMC_BAO(ucb.BAO_curved, paraminit, BAO_data, nstep, normal_vec, Prior2)


# SN + BAO data

# flat
def run_mcmc_flat_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., 4.66594018],[1, 36.39070654]])
    std = np.array([0.,  0.11162995])
    normal_vec = np.array([0.0003614, 0.11162995])*2.
    paraminit = np.array([0.9999742233029345,  31.689002893196257]) + np.random.normal(0,std)
    nstep = int(1.5*1e5)
    return ufsb.MCMC_SNBAO(ufsb.Other_stuff_flat, ufsb.BAO_flat, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)

# curved
def run_mcmc_curved_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., -3., 4.66594018],[20, 1., 35.39070654]])
    std = np.array([0.0790695, 0.13510082, 0.17355321])
    normal_vec = np.array([0.0590695, 0.08, 0.11])*1.5
    paraminit = np.array([3.518803845192434,  -0.19403381170893674,20.071034556246392]) + np.random.normal(0,std)
    nstep = int(1.5e5)
    return ucsb.MCMC_SNBAO(ucsb.Other_stuff_curved, ucsb.BAO_curved, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)

# execution
n_chain = 4
# SN flat
"""
print("Running MCMC for SN flat...")
with mp.Pool(processes=n_chain) as pool:
    results1 = pool.map(run_mcmc_flat_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_flat_SN_{i}.npy', results1[i])
"""
"""
# SN curved
print("Running MCMC for SN curved...")
with mp.Pool(processes=n_chain) as pool:
    results2 = pool.map(run_mcmc_curved_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_curved_SN_{i}.npy', results2[i])
"""
# BAO flat
print("Running MCMC for BAO flat...")
with mp.Pool(processes=n_chain) as pool:
    results3 = pool.map(run_mcmc_flat_BAO, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_flat_BAO_{i}.npy', results3[i])

# BAO curved
print("Running MCMC for BAO curved...")
with mp.Pool(processes=n_chain) as pool:
    results4 = pool.map(run_mcmc_curved_BAO, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_curved_BAO_{i}.npy', results4[i])
"""
"""
# SN + BAO flat
print("Running MCMC for SN + BAO flat...")
with mp.Pool(processes=n_chain) as pool:
    results5 = pool.map(run_mcmc_flat_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_flat_SN+BAO_{i}.npy', results5[i])
"""

# SN + BAO curved
print("Running MCMC for SN + BAO curved...")
with mp.Pool(processes=n_chain) as pool:
    results6 = pool.map(run_mcmc_curved_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'{Field_research_path}/Results/MCMC/MCMC_curved_SN+BAO_{i}.npy', results6[i])
"""