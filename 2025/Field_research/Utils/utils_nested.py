from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils_flat as uf
import utils_curved as uc
import utils_flat_BAO as ufb
import utils_curved_BAO as ucb
import utils_flat_SNBAO as ufsb
import utils_curved_SNBAO as ucsb
import os


# read the SN data
current_file_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_file_path)
Field_research_dir = os.path.dirname(utils_dir)
sndata = pd.read_csv(Field_research_dir + '/Data/parsonage.txt', sep = ' ', engine='python')
# read the BAO data
BAO_data = pd.read_csv(Field_research_dir + '/Data/BAO_data.csv')

# Defining the Loglikelihoods and priors for each cosmology
# 1. SN only
# 1-1. SN flat cosmology
#   Loglikelihood for SN flat cosmology
def Loglikelihood_SN_flat(parm):
    loglikelihood_SN = uf.Loglikelihood(uf.Other_stuff_flat,parm)
    return loglikelihood_SN
#   prior for SN flat cosmology
def prior_SN_flat(x):
    # return Omegam in [0,1] for flat universe
    # x is a uniform distribution in [0,1]
    return x

# 1-2. SN curved cosmology
#   Loglikelihood for SN curved cosmology
def Loglikelihood_SN_curved(parm):
    loglikelihood_SN = uc.Loglikelihood(uc.Other_stuff_curved,parm)
    return loglikelihood_SN
#   prior for SN curved cosmology
def prior_SN_curved(x):
    # Omegam in [0,6.437185929648241], Omegalamb in [-3,1]
    # prior[0] = Omegam, prior[1] = Omegalamb
    prior = x * np.array([6.437185929648241,4]) + np.array([0,-3]) 
    return prior

# 2. BAO only
# 2-1. BAO flat cosmology
#  Loglikelihood for BAO flat cosmology
def Loglikelihood_BAO_flat(parm):
    loglikelihood_BAO = ufb.BAO_loglikelihood(ufb.BAO_flat, parm)
    return loglikelihood_BAO
def prior_BAO_flat(x):
    # Omegam in [0,1], H0 in [7.309670710, 33.74697601]
    # prior[0] = Omegam, prior[1] = H0
    prior = x * np.array([1, 33.74697601 - 7.309670710]) + np.array([0, 7.309670710])
    return prior
# 2-2. BAO curved cosmology
#  Loglikelihood for BAO curved cosmology
def Loglikelihood_BAO_curved(parm):
    loglikelihood_BAO = ucb.BAO_loglikelihood(ucb.BAO_curved, parm)
    return loglikelihood_BAO
def prior_BAO_curved(x):
    # Omegam in [0,6.437185929648241], Omegalamb in [-3,1], H0 in [7.309670710, 23.17205389]
    prior = x * np.array([6.437185929648241, 4, 23.17205389 - 7.309670710]) + np.array([0, -3, 7.309670710])
    return prior

# 3. SN + BAO
# 3-1. SN + BAO flat cosmology
#   Loglikelihood for SN + BAO flat cosmology
def Loglikelihood_SN_BAO_flat(parm):
    loglikelihood_SN = ufsb.SN_Loglikelihood(ufsb.Other_stuff_flat,parm[:1])
    loglikelihood_BAO = ufsb.BAO_loglikelihood(ufsb.BAO_flat, parm)
    return loglikelihood_SN + loglikelihood_BAO
#   prior for SN + BAO flat cosmology
def prior_SN_BAO_flat(x):
    # Omegam in [0,1],H0 in [7.309670710, 33.74697601]
    # prior[0] = Omegam, prior[1] = H0
    prior = x * np.array([1, 33.74697601 - 7.309670710]) + np.array([0, 7.309670710])
    return prior

# 3-2. SN + BAO curved cosmology
#   Loglikelihood for SN + BAO curved cosmology
def Loglikelihood_SN_BAO_curved(parm):
    loglikelihood_SN = ucsb.SN_Loglikelihood(ucsb.Other_stuff_curved,parm[:2])
    loglikelihood_BAO = ucsb.BAO_loglikelihood(ucsb.BAO_curved, parm)
    return loglikelihood_SN + loglikelihood_BAO
#   prior for SN + BAO curved cosmology
def prior_SN_BAO_curved(x):
    # Omegam in [0,6.437185929648241], Omegalamb in [-3,1], H0 in [7.309670710, 23.17205389]
    prior = x * np.array([6.437185929648241, 4, 23.17205389 - 7.309670710]) + np.array([0, -3, 7.309670710])
    return prior
