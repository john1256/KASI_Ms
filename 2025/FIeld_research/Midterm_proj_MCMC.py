import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import multiprocess as mp
import utils_flat
import utils_flat_BAO
import utils_curved_BAO
import utils_curved


sys.path.append('./2025/Field_research/Utils')

CurrentPath = os.getcwd()
sndata = pd.read_csv(CurrentPath + '/2025/Field_research/Data/parsonage.txt', sep = ' ', engine='python')
# BAO data
BAO_z = np.array([0.01115,0.157,0.389,0.389,0.536,0.536,0.626,0.626])
BAO_val = np.array([0.08,1906.18,3905.35,43.56,4761.80,50.83,5175.36,57.19])
BAO_err = np.array([0.003,66.577,58.667,1.310,68.866,1.531,77.639,1.336])
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
    return utils_flat.MCMC(utils_flat.Other_stuff_flat, paraminit, sndata, nstep, normal_vec, Prior1)

# curved
def run_mcmc_curved_SN(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., 0.],[4.5, 4.5]])
    std = np.array([0.07765663, 0.05880332])
    normal_vec = np.array([1., 0.2])
    paraminit = np.array([3.5135883019039014,  0.002435561190052171]) + np.random.normal(0,std)
    nstep = int(1e6)
    return utils_curved.MCMC(utils_curved.Other_stuff_curved, paraminit, sndata, nstep, normal_vec, Prior2)

# SN + BAO data
# flat
def run_mcmc_flat_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., 4.66594018],[1, 36.39070654]])
    std = np.array([0., 0.57213598])
    normal_vec = np.array([0.0003, 0.11904393])*0.5
    paraminit = np.array([0.9999742233029345,  22.850391597263012]) + np.random.normal(0,std)
    nstep = int(2e5)
    return utils_flat_BAO.MCMC_BAO(utils_flat_BAO.Other_stuff_flat, utils_flat_BAO.BAO_flat, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)


# curved
def run_mcmc_curved_SNBAO(seed):
    np.random.seed(seed)
    Prior2 = np.array([[0., 0., 0.],[2.5, 2.5, 36.39070654]])
    std = np.array([0., 0., 0.57213598])
    normal_vec = np.array([0.0003, 0.11904393, 0.]) * 0.5
    paraminit = np.array([0.9999742233029345, 22.850391597263012, 1]) + np.random.normal(0,std)
    nstep = int(2e5)
    return utils_curved_BAO.MCMC_BAO(utils_curved_BAO.Other_stuff_curved, utils_curved_BAO.BAO_curved, paraminit, sndata,BAO_data, nstep, normal_vec, Prior2)


# execution    
n_chain = 4
# SN flat
with mp.Pool(processes=n_chain) as pool:
    results1 = pool.map(run_mcmc_flat_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/FIeld_research/Results/MCMC_flat_SN_{i}.npy', results1[i])
# SN curved
with mp.Pool(processes=n_chain) as pool:
    results2 = pool.map(run_mcmc_curved_SN, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/FIeld_research/Results/MCMC_curved_SN_{i}.npy', results2[i])

# SN + BAO flat
with mp.Pool(processes=n_chain) as pool:
    results3 = pool.map(run_mcmc_flat_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/FIeld_research/Results/MCMC_flat_SN+BAO_{i}.npy', results3[i])

# SN + BAO curved
with mp.Pool(processes=n_chain) as pool:
    results4 = pool.map(run_mcmc_curved_SNBAO, range(n_chain))
for i in range(n_chain):
    np.save(f'./2025/FIeld_research/Results/MCMC_curved_SN+BAO_{i}.npy', results4[i])