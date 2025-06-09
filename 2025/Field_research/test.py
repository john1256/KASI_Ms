import os
import numpy as np
CurrentPath = os.getcwd()
print(CurrentPath)
Filename = '2025/Field_research/Results/MCMC_flat_SN_0.npy'
np.load(Filename)
print("Test completed successfully.")
print(Filename)