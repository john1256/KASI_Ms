import numpy as np
from func import *
from scipy.interpolate import interp1d
import os

data_f = np.loadtxt(os.path.join(os.getcwd(), 'Assignment5', 'regression_data_f.txt'))
data_h = np.loadtxt(os.path.join(os.getcwd(), 'Assignment5', 'regression_data_h.txt'))
data_j = np.loadtxt(os.path.join(os.getcwd(), 'Assignment5', 'regression_data_j.txt'))
data_k = np.loadtxt(os.path.join(os.getcwd(), 'Assignment5', 'regression_data_k.txt'))

def M1(x, parm):
    return parm[0]*x + x**parm[1] + 1
def M2(x, parm):
    return parm[0] * np.tanh(x - parm[1]) + parm[2]
def M3(x, parm):
    return parm[0]*x * (np.sin(x) + parm[1]) + 1
def M4(x, parm):
    return parm[0] + parm[1]*(1 + x)
def M5(x, parm):
    return np.sqrt(parm[0] * (1 + x)**3 + parm[1])

theory = [M1, M2, M3, M4, M5]
theory2 = [M1_test, M2_test, M3_test, M4_test, M5_test]
data = [data_f, data_h, data_j, data_k]

for i in range(len(data)):
    xarr = data[i][:,0]
    yarr = data[i][:,1]
    sig_yarr = data[i][:,2]
    with open(f'best_fit_{i+1}.txt', 'w') as f:
        for j in range(len(theory)):
            print(f'Trying data {i+1} with model {j+1}')
            if j == 1:
                init_parm = set_init_point(theory2[j], data[i],3)
            else:
                init_parm = set_init_point(theory2[j], data[i],2)
            Th_parm, Th_chi2=Regression(theory[j], xarr, yarr, init_parm, sig_yarr, max_iter=1e10, learning_rate=1.)
            f.write(f'Model{j+1} {Th_parm} {Th_chi2}\n')
            
