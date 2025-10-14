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

def make_grids(*args):
    if len(args)==3:
        center_x, center_y, range = args
        x = np.linspace(center_x - range, center_x + range, 21)  # precision = 2*range / 20
        y = np.linspace(center_y - range, center_y + range, 21)
        return np.meshgrid(x, y)
    elif len(args)==4:
        center_x, center_y, center_z, range = args   
        z = np.linspace(center_z - range, center_z + range, 21)
        x = np.linspace(center_x - range, center_x + range, 21)  # precision = 2*range / 20
        y = np.linspace(center_y - range, center_y + range, 21)
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
 # precision : 10^-precision
    center_x, center_y,center_z = 0, 0,0
    if dof==2:
        for i in range(precision):
            X,Y = make_grids(center_x, center_y, 10**(-i+1))
            chi_sq = chi2_grid(func, data, X,Y)
            min_idx = chi_sq.argmin()
            chi_sq_min = chi_sq.flatten()[min_idx]
            center_x = X.flatten()[min_idx]
            center_y = Y.flatten()[min_idx]
        return  center_x, center_y, chi_sq_min
    elif dof==3:
        for i in range(precision):
            X,Y,Z = make_grids(center_x, center_y, center_z, 10**(-i+1))
            chi_sq = chi2_grid(func, data, X,Y,Z)
            min_idx = chi_sq.argmin()
            center_x = X.flatten()[min_idx]
            center_y = Y.flatten()[min_idx]
            center_z = Z.flatten()[min_idx]
            chi_sq_min = chi_sq.flatten()[min_idx]
        return center_x, center_y, center_z, chi_sq_min

def draw_contour(func, data, )