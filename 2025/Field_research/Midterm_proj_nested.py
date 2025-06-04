import numpy as np
import matplotlib.pyplot as plt
import nestle
import corner
import sys
sys.path.append('./Utils')
import utils_nested as un

# plot results
# 1. SN only data
# 1-1. flat cosmology
print("Sampling SN flat cosmology...")
result_SN_flat = nestle.sample(un.Loglikelihood_SN_flat, un.prior_SN_flat, 1, method='single', npoints=200)
np.save()
p, cov = nestle.mean_and_cov(result_SN_flat.samples, result_SN_flat.weights)

with open('Results/SN_flat_result.txt', 'w') as f:
    f.write(result_SN_flat.summary())
    f.write(f'\nlogzerr : {result_SN_flat.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, H0 = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')

fig = corner.corner(result_SN_flat.samples, weights=result_SN_flat.weights, labels=[r'$\Omega_m$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999], bins=30, show_titles=True)
_ = fig.suptitle('SN only flat LCDM cosmology nested sampling result', fontsize=16)
fig.tight_layout()
fig.savefig('./Figs/SN_flat_corner.png', dpi=300)

# 1-2. curved cosmology
print("Sampling SN curved cosmology...")
result_SN_curved = nestle.sample(un.Loglikelihood_SN_curved, un.prior_SN_curved, 2, method='single', npoints=200)
p, cov = nestle.mean_and_cov(result_SN_curved.samples, result_SN_curved.weights)
with open('Results/SN_curved_result.txt', 'w') as f:
    f.write(result_SN_curved.summary())
    f.write(f'\nlogzerr : {result_SN_curved.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, Omegalamb = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')
fig2 = corner.corner(result_SN_curved.samples, weights=result_SN_curved.weights, labels=[r'$\Omega_m$', r'$\Omega_\Lambda$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999], bins=30, show_titles=True)
_ = fig2.suptitle('SN only curved LCDM cosmology nested sampling result', fontsize=16)
fig2.tight_layout()
fig2.savefig('./Figs/SN_curved_corner.png', dpi=300)


# 2. SN + BAO data
# 2-1. flat cosmology
print("Sampling SN + BAO flat cosmology...")
result_SN_BAO_flat = nestle.sample(un.Loglikelihood_SN_BAO_flat, un.prior_SN_BAO_flat, 2, method='single', npoints=200)
p, cov = nestle.mean_and_cov(result_SN_BAO_flat.samples, result_SN_BAO_flat.weights)
with open('Results/SN_BAO_flat_result.txt', 'w') as f:
    f.write(result_SN_BAO_flat.summary())
    f.write(f'\nlogzerr : {result_SN_BAO_flat.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, H0 = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')
fig3 = corner.corner(result_SN_BAO_flat.samples, weights=result_SN_BAO_flat.weights, labels=[r'$\Omega_m$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999], bins=30, show_titles=True)
_ = fig3.suptitle('SN + BAO flat LCDM cosmology nested sampling result', fontsize=16)
fig3.tight_layout()
fig3.savefig('./Figs/SN_BAO_flat_corner.png', dpi=300)

# 2-2. curved cosmology
print("Sampling SN + BAO curved cosmology...")
result_SN_BAO_curved = nestle.sample(un.Loglikelihood_SN_BAO_curved, un.prior_SN_BAO_curved, 3, method='single', npoints=200)
p, cov = nestle.mean_and_cov(result_SN_BAO_curved.samples, result_SN_BAO_curved.weights)
with open('Results/SN_BAO_curved_result.txt', 'w') as f:
    f.write(result_SN_BAO_curved.summary())
    f.write(f'\nlogzerr : {result_SN_BAO_curved.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, Omegalamb = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}, H0 = {p[2]:.3f} +/- {np.sqrt(cov[2,2]):.3f}')
fig4 = corner.corner(result_SN_BAO_curved.samples, weights=result_SN_BAO_curved.weights, labels=[r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999, 0.99999], bins=30, show_titles=True)
_ = fig4.suptitle('SN + BAO curved LCDM cosmology nested sampling result', fontsize=16)
fig4.tight_layout()
fig4.savefig('./Figs/SN_BAO_curved_corner.png', dpi=300)
