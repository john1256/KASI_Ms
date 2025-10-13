import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def mean(data):
    return np.sum(data)/len(data)
def geom_mean(data):
    if np.any(data<=0):
        return np.nan
    else:
        return np.exp(np.sum(np.log(data))/len(data))
def median(data):
    sort = np.sort(data)
    n = len(data)
    if n%2==0:
        return (sort[n//2-1]+sort[n//2])/2
    else:
        return sort[n//2]
def variance(data):
    return mean(data**2)-mean(data)**2
def std(data):
    return np.sqrt(variance(data))
def skewness(data):
    x3 = mean(data**3)
    x2 = mean(data**2)
    x = mean(data)
    sig = std(data)
    return 1/(sig**3)*(x3 - 3*x*x2 + 2*x**3)
def kurtosis(data):
    mu = mean(data)
    return mean((data - mu)**4) / (variance(data)**2) - 3
def weighted_mean(data, stds):
    weights = 1/stds**2
    return np.sum(data*weights)/np.sum(weights)
def weighted_std(stds):
    weights = 1/stds**2
    return np.sqrt(1/np.sum(weights))

def Chisq_pdf(chisq, n):
    chi = np.sqrt(chisq)
    prob = 2**(-n/2)/gamma(n/2)*chi**(n-2)*np.exp(-chisq/2)
    return prob
def Chisq_integral(chisq, n, to_inf=True):
    if to_inf:
        result, _ = quad(Chisq_pdf, chisq, np.inf, args = (n,))
    else:
        result, _ = quad(Chisq_pdf, 0, chisq, args = (n,))
    return result
# assignment 5 - 2
def _gradient(func, parm, x, h=1e-9):
    grad = []
    for i in range(len(parm)):
        parm_prime = parm.copy()
        parm_prime[i] += h
        f1 = func(x, parm)
        f_prime = func(x, parm_prime)
        grad_i = (f_prime - f1)/h
        grad.append(grad_i)
    grad = np.array(grad)
    return grad
def chi2(func, x, y, sig_y, parm):
    model = func(x, parm)
    chi2 = np.sum(((y - model)/sig_y)**2)
    return chi2
def _chi(func, x, y, sig_y, parm):
    model = func(x, parm)
    chi = (y - model)/sig_y
    return chi

def _grad_like(func, x, y, sig_y, parm):
    grad = _gradient(func, parm, x)
    chi = _chi(func, x, y, sig_y, parm)
    grad_like = []
    for gr in grad:
        grad_like.append(np.sum(gr*chi/sig_y))
    grad_like = -np.array(grad_like)
    return grad_like    

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



def set_init_grid(model_test, data, parmnum):
    if parmnum == 2:
        A, B = np.linspace(-9,9,10), np.linspace(-9,9,10)
        Agrid, Bgrid = np.meshgrid(A, B)
        perturb = np.random.uniform(-0.1,0.1, size=Agrid.shape)
        
        Agrid += perturb; Bgrid += perturb

        chi2_values = chi2_grid(model_test, data, Agrid, Bgrid)
        notnan = ~np.isnan(chi2_values)
        chi2_nan = chi2_values[notnan]; Agrid_nan = Agrid[notnan]; Bgrid_nan = Bgrid[notnan]
        sort_indices = np.argsort(chi2_nan)
        sorted_Agrid = Agrid_nan[sort_indices]
        sorted_Bgrid = Bgrid_nan[sort_indices]
 
        final_parm = np.array([sorted_Agrid, sorted_Bgrid]).T
        random_idx = np.random.randint(0,9)
        parm = final_parm[random_idx]
        final_choice = [np.random.uniform(parm[0]-0.9, parm[0]+0.9), np.random.uniform(parm[1]-0.9, parm[1]+0.9)]
    elif parmnum == 3:
        A, B, C = np.linspace(-9,9,10), np.linspace(-9,9,10), np.linspace(-9,9,10)
        Agrid, Bgrid, Cgrid = np.meshgrid(A, B, C)
        perturb = np.random.uniform(-0.1,0.1, size=Agrid.shape)
        Agrid += perturb; Bgrid += perturb; Cgrid += perturb
        chi2_values = chi2_grid(model_test, data, Agrid, Bgrid, Cgrid)
        notnan = ~np.isnan(chi2_values)
        chi2_nan = chi2_values[notnan]; Agrid_nan = Agrid[notnan]; Bgrid_nan = Bgrid[notnan]; Cgrid_nan = Cgrid[notnan]
        sort_indices = np.argsort(chi2_nan)
        sorted_Agrid = Agrid_nan[sort_indices]
        sorted_Bgrid = Bgrid_nan[sort_indices]
        sorted_Cgrid = Cgrid_nan[sort_indices]

        final_parm = np.array([sorted_Agrid, sorted_Bgrid, sorted_Cgrid]).T
        random_idx = np.random.randint(0,9)
        parm = final_parm[random_idx]
        final_choice = [np.random.uniform(parm[0]-0.9, parm[0]+0.9), np.random.uniform(parm[1]-0.9, parm[1]+0.9), np.random.uniform(parm[2]-0.9, parm[2]+0.9)]

    return np.array(final_choice)

def Regression(func,func_test, data, parmnum,max_iter=1000, learning_rate=1.):
    go=True
    init_parm = set_init_grid(func_test, data, parmnum)
    x = data[:,0]
    y = data[:,1]
    sig_y = data[:,2]
    print("Initial parameters:", init_parm)
    init_grad_like = _grad_like(func, x, y, sig_y, init_parm)
    init_grad_like_mag = np.sqrt(np.sum(init_grad_like**2))
    norm = 1/ init_grad_like_mag
    scale = int(np.floor(np.log10(norm)))
    learning_rate *= 10**scale
    old_parm = init_parm.copy()
    perturb_count = 0
    parm_history = []
    chi2_history = []
    while go==True and max_iter > 0:
        if (old_parm <=-10).all() and (old_parm >=10).all():
            print('out of bounds, perturbing...')
            old_parm = set_init_grid(func_test, data, parmnum)
        max_iter -= 1
        try: # chi2_new should be smaller than chi2_old
            chi2_old = chi2(func, x, y, sig_y, old_parm)
            if np.isnan(chi2_old):
                raise FloatingPointError("chi2_old is NaN")
            grad_like = _grad_like(func, x, y, sig_y, old_parm)
            new_parm = old_parm - learning_rate * grad_like
            chi2_new = chi2(func, x, y, sig_y, new_parm)
            if chi2_old - chi2_new < 1e-4: # chi2_new is worse than chi2_old
                parm_history.append(old_parm)
                chi2_history.append(chi2_old)
                if perturb_count > 50:
                    print("Perturbation limit reached.")
                    go = False
                    continue
                #perturb_amount = np.random.uniform(-1., 1., size=len(old_parm))
                #old_parm = np.clip(old_parm + perturb_amount, -10, 10)
                perturb_parm = set_init_grid(func_test, data, parmnum)
                old_parm = perturb_parm
                perturb_count += 1
            else:
                old_parm = new_parm
        except (ValueError, FloatingPointError) as e:
            print(f"Calculation error ({e}). Perturbing...")
            new_val = set_init_grid(func_test, data, parmnum)
            old_parm = new_val
        if max_iter ==0:
            print("Maximum iteration reached.")
    if chi2_history:
        parm_history_arr = np.array(parm_history)
        chi2_history_arr = np.array(chi2_history)
        
        is_not_nan_mask = ~np.isnan(chi2_history_arr)
        clean_parm_history = parm_history_arr[is_not_nan_mask]
        clean_chi2_history = chi2_history_arr[is_not_nan_mask]
        min_chi2_in_history = min(chi2_history)
        # 현재 chi2_old와 history의 최소값을 비교
        if min_chi2_in_history < chi2_old and clean_parm_history.size > 0:
            print("Returning best result from history.")
            min_idx = np.argmin(chi2_history)
            # grad_like는 마지막 상태의 값을 반환하거나, 필요하다면 다시 계산해야 함
            # 여기서는 간단히 None으로 처리하거나 마지막 grad_like를 반환
            return parm_history[min_idx], min_chi2_in_history
        else:
            print("Error.")
    return old_parm, chi2_old


# Assignment 5 - 3
def read_model_result(file_path, model_name="Model1"):
    """
    file_path: 텍스트 파일 경로
    model_name: 예) "Model1", "Model2", ...
    반환: (params: np.ndarray, chi2: float)
    """
    # 예: "Model1 [0.2312297  0.94008334] 96.07631173773878"
    import re
    pat = re.compile(rf'^{re.escape(model_name)}\s*\[([^\]]+)\]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$')
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if m:
                params_str = m.group(1)
                chi2_str = m.group(2)
                params = np.fromstring(params_str, sep=' ')
                chi2 = float(chi2_str)
                return params, chi2
    raise ValueError(f"{model_name} 라인을 파일에서 찾지 못했습니다: {file_path}")


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

def make_grid(span, steps=500):
    grid = np.logspace(0, np.log10(span), steps)-1
    grid_minus = -grid[::-1][:-1]
    grid_total = np.concatenate((grid_minus, grid))
    return grid_total

def plot_contour(*args, **kwargs):
    filename, modelname, grid_total, data = args
    plot_parms = kwargs['plot_parms'] if 'plot_parms' in kwargs else 'ab'
    dataname = kwargs['data_name'] if 'data_name' in kwargs else 'data_h'
    parms, _ = read_model_result(filename, modelname)
    model = [M1_test, M2_test, M3_test, M4_test, M5_test]
    if modelname == "Model1":
        model = M1_test
    elif modelname == "Model2":
        model = M2_test
    elif modelname == "Model3":
        model = M3_test
    elif modelname == "Model4":
        model = M4_test
    elif modelname == "Model5":
        model = M5_test
        
    from scipy.stats import chi2    
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    colors = ['green', 'blue', 'red']
    conf_lvl = ['68%', '95%', '99%']
    
    if len(parms) == 2:
        print(f'Initial parameters: {parms}')
        Agrid = grid_total[0] + parms[0]
        Bgrid = grid_total[1] + parms[1]
        Amesh, Bmesh = np.meshgrid(Agrid, Bgrid)
        chi2_vals = chi2_grid(model, data, Amesh, Bmesh)
        delta_chi2 = [chi2.ppf(0.68, 2), chi2.ppf(0.95, 2), chi2.ppf(0.99, 2)]
        min_chi2 = np.min(chi2_vals)
        min_id = (chi2_vals == min_chi2)
        parms_new = [Amesh[min_id][0], Bmesh[min_id][0]]
        print(f'New parameters: {parms_new}')
        print(f'Min chi2: {min_chi2}')
        fig, ax = plt.subplots(figsize=(7,6))
        for i in range(len(delta_chi2)):
            if i==0:
                chi2_bool = chi2_vals < min_chi2 + delta_chi2[i]
            else:
                chi2_bool = (chi2_vals > delta_chi2[i-1]) & (chi2_vals < min_chi2 + delta_chi2[i])
            A_masked = Amesh[chi2_bool]; B_masked = Bmesh[chi2_bool]; z_masked = chi2_vals[chi2_bool]
            np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{i}.npy', np.array([A_masked, B_masked, z_masked]))
            pts = np.c_[A_masked, B_masked]
            hull = ConvexHull(pts)
            V = pts[hull.vertices]
            ax.fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
        ax.scatter(parms_new[0], parms_new[1], color='black', marker='x', s=100, label=f'Best fit={parms_new[0]:.3f}, {parms_new[1]:.3f}, chi2 = {min_chi2:.3f}')
        ax.set_xlabel('a'); ax.set_ylabel('b')
        ax.legend()
        ax.set_title(f'{modelname}, {dataname}')
        plt.tight_layout()
        plt.savefig(f'./Assignment5/figs/contour_{modelname}_{dataname}')
        return min_chi2, parms_new
    elif len(parms) == 3:
        print(f'Initial parameters: {parms}')
        Amesh = grid_total[0] + parms[0]
        Bmesh = grid_total[1] + parms[1]
        Cmesh = grid_total[2] + parms[2]
        X,Y,Z = np.meshgrid(Amesh, Bmesh, Cmesh)
        chi2_vals = chi2_grid(model, data, X, Y, Z)
        new_chi2_min= np.min(chi2_vals)
        id = (chi2_vals == new_chi2_min)
        print(f'Min chi2: {new_chi2_min}')
        Xnew = X[id]; Ynew = Y[id]; Znew = Z[id]
        parms_new = [Xnew[0], Ynew[0], Znew[0]]
        print(f'New parameters: {parms_new}')
        delta_chi2_full = [chi2.ppf(0.68, 3), chi2.ppf(0.95, 3), chi2.ppf(0.99, 3)]
        print('saving full 3d contours')
        for i in range(len(delta_chi2_full)):
            if i==0:
                chi2_bool = chi2_vals < new_chi2_min + delta_chi2_full[i]
            else:
                chi2_bool = (chi2_vals > delta_chi2_full[i-1]) & (chi2_vals < new_chi2_min + delta_chi2_full[i])
            X_masked = X[chi2_bool]; Y_masked = Y[chi2_bool]; Z_masked = Z[chi2_bool]; chi2_masked = chi2_vals[chi2_bool]
            np.save(f'./Assignment5/data/cont_{modelname}_{dataname}_{i}.npy', np.array([X_masked, Y_masked, Z_masked, chi2_masked]))
        Agrid = grid_total[0] + parms_new[0]
        Bgrid = grid_total[1] + parms_new[1]
        Cgrid = grid_total[2] + parms_new[2]
        Abmesh, aBmesh = np.meshgrid(Agrid, Bgrid)
        Acmesh, aCmesh = np.meshgrid(Agrid, Cgrid)
        Bcmesh, bCmesh = np.meshgrid(Bgrid, Cgrid)
        chi2_abvals = chi2_grid(model, data, Abmesh, aBmesh, parms_new[2])
        chi2_acvals = chi2_grid(model, data, Acmesh, aCmesh, parms_new[1])
        chi2_bcvals = chi2_grid(model, data, Bcmesh, bCmesh, parms_new[0])
        min_chi2_ab = np.min(chi2_abvals)
        min_chi2_ac = np.min(chi2_acvals)
        min_chi2_bc = np.min(chi2_bcvals)
        delta_chi2 = [chi2.ppf(0.68, 2), chi2.ppf(0.95, 2), chi2.ppf(0.99, 2)]
            # 1. a-b
        if plot_parms == 'ab':
            fig, ax = plt.subplots(figsize=(7,6))
            for i in range(len(delta_chi2)):
                chi2_ab_bool = chi2_abvals < min_chi2_ab + delta_chi2[i]
                Ab_masked = Abmesh[chi2_ab_bool]; aB_masked = aBmesh[chi2_ab_bool]
                pts = np.c_[Ab_masked, aB_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax.fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
            ax.scatter(parms_new[0], parms_new[1], color='black', marker='x', s=100, label=f'Best fit={parms_new[0]:.3f}, {parms_new[1]:.3f}, chi2 = {new_chi2_min:.3f}')
            ax.set_xlabel('a'); ax.set_ylabel('b')
            ax.set_title(f'{modelname}, {dataname} (a-b)')
            ax.legend()
        # 2. b-c
        elif plot_parms == 'bc':
            fig, ax = plt.subplots(figsize=(7,6))
            for i in range(len(delta_chi2)):
                chi2_bc_bool = chi2_bcvals < min_chi2_bc + delta_chi2[i]
                Bc_masked = Bcmesh[chi2_bc_bool]; bC_masked = bCmesh[chi2_bc_bool]
                pts = np.c_[Bc_masked, bC_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax.fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
            ax.scatter(parms_new[1], parms_new[2], color='black', marker='x', s=100, label=f'Best fit={parms_new[1]:.3f}, {parms_new[2]:.3f}, chi2 = {min_chi2_bc:.3f}')
            ax.set_xlabel('b'); ax.set_ylabel('c')
            ax.set_title(f'{modelname}, {dataname} (b-c)')
            ax.legend()
        # 3. a-c
        elif plot_parms == 'ac':
            fig, ax = plt.subplots(figsize=(7,6))
            for i in range(len(delta_chi2)):
                chi2_ac_bool = chi2_acvals < min_chi2_ac + delta_chi2[i]
                Ac_masked = Acmesh[chi2_ac_bool]; aC_masked = aCmesh[chi2_ac_bool]
                pts = np.c_[Ac_masked, aC_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax.fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
            ax.scatter(parms_new[0], parms_new[2], color='black', marker='x', s=100, label=f'Best fit={parms_new[0]:.3f}, {parms_new[2]:.3f}, chi2 = {min_chi2_ac:.3f}')
            ax.set_xlabel('a'); ax.set_ylabel('c')
            ax.set_title(f'{modelname}, {dataname} (a-c)')
            ax.legend()
        # all
        elif plot_parms == 'abc':
            fig, ax = plt.subplots(1,3, figsize=(18,5))
            for i in range(len(delta_chi2)):
                # 1. a-b
                chi2_ab_bool = chi2_abvals < min_chi2_ab + delta_chi2[i]
                Ab_masked = Abmesh[chi2_ab_bool]; aB_masked = aBmesh[chi2_ab_bool]
                pts = np.c_[Ab_masked, aB_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax[0].fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
                # 2. b-c
                chi2_bc_bool = chi2_bcvals < min_chi2_bc + delta_chi2[i]
                Bc_masked = Bcmesh[chi2_bc_bool]; bC_masked = bCmesh[chi2_bc_bool]
                pts = np.c_[Bc_masked, bC_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax[1].fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
                # 3. a-c
                chi2_ac_bool = chi2_acvals < min_chi2_ac + delta_chi2[i]
                Ac_masked = Acmesh[chi2_ac_bool]; aC_masked = aCmesh[chi2_ac_bool]
                pts = np.c_[Ac_masked, aC_masked]
                hull = ConvexHull(pts)
                V = pts[hull.vertices]
                ax[2].fill(V[:,0], V[:,1], facecolor='none', lw=2, edgecolor=colors[i], label=conf_lvl[i])
            # 1. a-b
            ax[0].scatter(parms_new[0], parms_new[1], color='black', marker='x', s=100, label=f'Best fit={parms_new[0]:.3f}, {parms_new[1]:.3f}, chi2 = {min_chi2_ab:.3f}')
            ax[0].set_xlabel('a'); ax[0].set_ylabel('b')
            ax[0].set_title(f'{modelname}, {dataname} (a-b)')
            ax[0].legend()
            # 2. b-c
            ax[1].scatter(parms_new[1], parms_new[2], color='black', marker='x', s=100, label=f'Best fit={parms_new[1]:.3f}, {parms_new[2]:.3f}, chi2 = {min_chi2_bc:.3f}')
            ax[1].set_xlabel('b'); ax[1].set_ylabel('c')
            ax[1].set_title(f'{modelname}, {dataname} (b-c)')
            ax[1].legend()
            # 3. a-c
            ax[2].scatter(parms_new[0], parms_new[2], color='black', marker='x', s=100, label=f'Best fit={parms_new[0]:.3f}, {parms_new[2]:.3f}, chi2 = {min_chi2_ac:.3f}')
            ax[2].set_xlabel('a'); ax[2].set_ylabel('c')
            ax[2].set_title(f'{modelname}, {dataname} (a-c)')
            ax[2].legend()

        plt.tight_layout()
        plt.savefig(f'./Assignment5/figs/contour_{modelname}_{dataname}')
        return new_chi2_min, parms_new