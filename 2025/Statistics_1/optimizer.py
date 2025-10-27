import numpy as np

def M1_test(x, a, b):
    return a*x + x**b + 1
def M2_test(x, a, b, c):
    return a * np.tanh(x - b) + c
def M3_test(x, a, b):
    return a*x * (np.sin(x) + b) + 1
def M4_test(x, a, b):
    return a + b*(1 + x)
def M5_test(x, a, b):
    return np.sqrt(a * (1 + x)**3 + b)

def make_grids(*args, **kwargs):
    num_points = kwargs['num_points'] if 'num_points' in kwargs else 21
    if len(args)==3:
        center_x, center_y, range = args
        x = np.linspace(center_x - range[0], center_x + range[0], num_points)  # precision = 2*range / 20
        y = np.linspace(center_y - range[1], center_y + range[1], num_points)
        return np.meshgrid(x, y)
    elif len(args)==4:
        center_x, center_y, center_z, range = args
        z = np.linspace(center_z - range[2], center_z + range[2], num_points)
        x = np.linspace(center_x - range[0], center_x + range[0], num_points)  # precision = 2*range / 20
        y = np.linspace(center_y - range[1], center_y + range[1], num_points)
        return np.meshgrid(x, y, z)

def chi2_grid(*args):
    chi2_grid_val = 0
    if len(args) == 4:
        model, data, a, b = args
    elif len(args) == 5:
        model, data, a, b, c = args
    for d in data:
        x,y,sig = d
        if len(args) == 4:
            chi2_grid_val += ((y - model(x, a, b))/sig)**2
        elif len(args) == 5:
            chi2_grid_val += ((y - model(x, a, b, c))/sig)**2
    return chi2_grid_val

def chisq_min(func, data, dof, precision=3):
    print(f'precision : 1e{-precision}')
    center_x, center_y,center_z = 0, 0,0
    if dof==2:
        for i in range(precision+1):
            X,Y = make_grids(center_x, center_y, [(i+1)*10**(-i+1),(i+1)*10**(-i+1)], num_points = 81)
            chi_sq = chi2_grid(func, data, X,Y)
            min_idx = np.nanargmin(chi_sq)
            chi_sq_min = chi_sq.flatten()[min_idx]
            center_x = X.flatten()[min_idx]
            center_y = Y.flatten()[min_idx]
        return  center_x, center_y, chi_sq_min
    elif dof==3:
        for i in range(precision+1):
            X,Y,Z = make_grids(center_x, center_y, center_z, [10**(-i+1),10**(-i+1),2*10**(-i+1)], num_points = 81)
            chi_sq = chi2_grid(func, data, X,Y,Z)
            min_idx = np.nanargmin(chi_sq)
            center_x = X.flatten()[min_idx]
            center_y = Y.flatten()[min_idx]
            center_z = Z.flatten()[min_idx]
            chi_sq_min = chi_sq.flatten()[min_idx]
        return center_x, center_y, center_z, chi_sq_min

def draw_contour(func, data, best_fit, grid_range, num_points = 50, **kwargs):
    import matplotlib.pyplot as plt
    modelname = kwargs['modelname'] if 'modelname' in kwargs else 'Model1'
    dataname = kwargs['dataname'] if 'dataname' in kwargs else 'data_f'
    from scipy.stats import chi2
    if len(best_fit) == 2:
        X,Y = make_grids(best_fit[0], best_fit[1], grid_range, num_points=num_points)
        chi_sq = chi2_grid(func, data, X,Y)
        chi_sq_min = chi_sq.min()
        conf_interv = [chi2.ppf(0.68, df=2), chi2.ppf(0.95, df=2), chi2.ppf(0.99, df=2)]
        _, ax = plt.subplots(figsize=(8,6))
        for i in [2,1,0]:
            if i==0:
                chi_sq_bool = chi_sq < chi_sq_min + conf_interv[i]
                X_conf = X[chi_sq_bool]
                Y_conf = Y[chi_sq_bool]
                parm_arr = np.vstack((X_conf, Y_conf)).T
                CI = np.random.choice(parm_arr.shape[0], size=800, replace =False)
                parm_arr = parm_arr[CI]
                np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{i}.npy', parm_arr)
            else:
                chi_sq_bool = (chi_sq > chi_sq_min + conf_interv[i-1]) & (chi_sq < chi_sq_min + conf_interv[i])
                X_conf = X[chi_sq_bool]
                Y_conf = Y[chi_sq_bool]
                parm_arr = np.vstack((X_conf, Y_conf)).T
                CI = np.random.choice(parm_arr.shape[0], size=800, replace =False)
                parm_arr = parm_arr[CI]
                np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{i}.npy', parm_arr)
            
            ax.scatter(parm_arr[:, 0], parm_arr[:, 1], s=5, label = f'{[0.68, 0.95, 0.99][i]*100}% conf.', color=['g','b','r'][i])
        ax.set_xlabel('a')
        ax.set_ylabel('b')
            
    elif len(best_fit) == 3:
        X,Y,Z = make_grids(best_fit[0], best_fit[1], best_fit[2], grid_range, num_points=num_points)
        chi_sq = chi2_grid(func, data, X,Y,Z)
        chi_sq_min = chi_sq.min()
        conf_interv = [chi2.ppf(0.68, df=3), chi2.ppf(0.95, df=3), chi2.ppf(0.99, df=3)]
        _, ax = plt.subplots(1,3, figsize=(18,5))
        for i in [2,1,0]:
            if i==0:
                chi_sq_bool = chi_sq < chi_sq_min + conf_interv[i]
                X_conf = X[chi_sq_bool]
                Y_conf = Y[chi_sq_bool]
                Z_conf = Z[chi_sq_bool]
                parm_arr = np.vstack((X_conf, Y_conf, Z_conf)).T
                CI = np.random.choice(parm_arr.shape[0], size=800, replace =False)
                parm_arr = parm_arr[CI]
                np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{2-i}.npy', parm_arr)
            else:
                chi_sq_bool = (chi_sq > chi_sq_min + conf_interv[i-1]) & (chi_sq < chi_sq_min + conf_interv[i])
                X_conf = X[chi_sq_bool]
                Y_conf = Y[chi_sq_bool]
                Z_conf = Z[chi_sq_bool]
                parm_arr = np.vstack((X_conf, Y_conf, Z_conf)).T
                CI = np.random.choice(parm_arr.shape[0], size=800, replace =False)
                parm_arr = parm_arr[CI]
                np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{2-i}.npy', parm_arr)
            
            ax[0].scatter(parm_arr[:, 0], parm_arr[:, 1], s=5, label = f'{[0.68, 0.95, 0.99][i]*100}% conf.', color=['g','b','r'][i])
            ax[0].set_xlabel('a')
            ax[0].set_ylabel('b')
            ax[1].scatter(parm_arr[:, 0], parm_arr[:, 2], s=5, label = f'{[0.68, 0.95, 0.99][i]*100}% conf.', color=['g','b','r'][i])
            ax[1].set_xlabel('a')
            ax[1].set_ylabel('c')
            ax[2].scatter(parm_arr[:, 1], parm_arr[:, 2], s=5, label = f'{[0.68, 0.95, 0.99][i]*100}% conf.', color=['g','b','r'][i])
            ax[2].set_xlabel('b')
            ax[2].set_ylabel('c')
    plt.legend()
    plt.suptitle(f'Confidence intervals of {modelname} fitted to {dataname}')
    plt.tight_layout()
    plt.savefig(f'./Assignment5/figs/cont_{modelname}_{dataname}.png', dpi=300)
    plt.show()
