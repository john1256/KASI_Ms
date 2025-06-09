from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils_flat as uf
import utils_curved as uc
import utils_flat_BAO as ufb
import utils_curved_BAO as ucb
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
    loglikelihood_SN = uf.Loglikelihood(uf.Other_stuff_flat,parm, sndata)
    return loglikelihood_SN
#   prior for SN flat cosmology
def prior_SN_flat(x):
    # return Omegam in [0,1] for flat universe
    # x is a uniform distribution in [0,1]
    return x

# 1-2. SN curved cosmology
#   Loglikelihood for SN curved cosmology
def Loglikelihood_SN_curved(parm):
    loglikelihood_SN = uc.Loglikelihood(uc.Other_stuff_curved,parm, sndata)
    return loglikelihood_SN
#   prior for SN curved cosmology
def prior_SN_curved(x):
    # Omegam in [0,20], Omegalamb in [-3,3]
    # prior[0] = Omegam, prior[1] = Omegalamb
    prior = x * np.array([10,6]) + np.array([0,-3]) 
    return prior

# 2. SN + BAO
# 2-1. SN + BAO flat cosmology
#   Loglikelihood for SN + BAO flat cosmology
def Loglikelihood_SN_BAO_flat(parm):
    loglikelihood_SN = ufb.SN_Loglikelihood(ufb.Other_stuff_flat,parm[:1], sndata)
    loglikelihood_BAO = ufb.BAO_loglikelihood(ufb.BAO_flat, parm, BAO_data)
    return loglikelihood_SN + loglikelihood_BAO
#   prior for SN + BAO flat cosmology
def prior_SN_BAO_flat(x):
    # Omegam in [0,1],H0 in [4.66594018, 36.39070654]
    # prior[0] = Omegam, prior[1] = H0
    prior = x * np.array([1, 31.72476636]) + np.array([0, 4.66594018])
    return prior

# 2-2. SN + BAO curved cosmology
#   Loglikelihood for SN + BAO curved cosmology
def Loglikelihood_SN_BAO_curved(parm):
    loglikelihood_SN = ucb.SN_Loglikelihood(ucb.Other_stuff_curved,parm[:2], sndata)
    loglikelihood_BAO = ucb.BAO_loglikelihood(ucb.BAO_curved, parm, BAO_data)
    return loglikelihood_SN + loglikelihood_BAO
#   prior for SN + BAO curved cosmology
def prior_SN_BAO_curved(x):
    # Omegam in [0,20], Omegalamb in [-1.6,1.6], H0 in [4.66594018, 35.39070654]
    prior = x * np.array([20, 3.2, 35.39070654 - 4.66594018]) + np.array([0, -1.6, 4.66594018])
    return prior
