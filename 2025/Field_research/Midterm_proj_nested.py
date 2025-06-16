import numpy as np
import matplotlib.pyplot as plt
import nestle
import corner
import sys
import pickle
import os

Field_research_dir = os.path.dirname(__file__)
sys.path.append(Field_research_dir + '/Utils')
import utils_nested as un

# plot results
# 1. SN only data
# 1-1. flat cosmology

print("Sampling SN flat cosmology...")
result_SN_flat = nestle.sample(un.Loglikelihood_SN_flat, un.prior_SN_flat, 1, method='single', npoints=200, callback = nestle.print_progress)
# saving the result
print("Saving SN flat cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_SN_flat_result.pkl', 'wb') as f:
    pickle.dump(result_SN_flat, f)
p, cov = nestle.mean_and_cov(result_SN_flat.samples, result_SN_flat.weights)

with open(Field_research_dir + '/Results/Nested/nested_SN_flat_summary.txt', 'w') as f:
    f.write(result_SN_flat.summary())
    f.write(f'\nlogzerr : {result_SN_flat.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}')

fig = corner.corner(result_SN_flat.samples, weights=result_SN_flat.weights, labels=[r'$\Omega_m$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999], bins=30, show_titles=True)
fig.tight_layout()
fig.savefig(Field_research_dir + '/Figs/nested_SN_flat.png', dpi=300)


# 1-2. curved cosmology
print("Sampling SN curved cosmology...")
result_SN_curved = nestle.sample(un.Loglikelihood_SN_curved, un.prior_SN_curved, 2, method='single', npoints=200, callback = nestle.print_progress)
# saving the result
print("Saving SN curved cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_SN_curved_result.pkl', 'wb') as f:
    pickle.dump(result_SN_curved, f)
p, cov = nestle.mean_and_cov(result_SN_curved.samples, result_SN_curved.weights)
with open(Field_research_dir + '/Results/Nested/nested_SN_curved_summary.txt', 'w') as f:
    f.write(result_SN_curved.summary())
    f.write(f'\nlogzerr : {result_SN_curved.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, Omegalamb = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')
fig2 = corner.corner(result_SN_curved.samples, weights=result_SN_curved.weights, labels=[r'$\Omega_m$', r'$\Omega_\Lambda$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999], bins=30, show_titles=True)
fig2.tight_layout()
fig2.savefig(Field_research_dir + '/Figs/nested_SN_curved.png', dpi=300)

# 2. BAO only data
# 2-1. flat cosmology
print("Sampling BAO flat cosmology...")
result_BAO_flat = nestle.sample(un.Loglikelihood_BAO_flat, un.prior_BAO_flat, 2, method='single', npoints=200, callback = nestle.print_progress)
# saving the result
print("Saving BAO flat cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_BAO_flat_result.pkl', 'wb') as f:
    pickle.dump(result_BAO_flat, f)
p, cov = nestle.mean_and_cov(result_BAO_flat.samples, result_BAO_flat.weights)
with open(Field_research_dir + '/Results/Nested/nested_BAO_flat_summary.txt', 'w') as f:
    f.write(result_BAO_flat.summary())
    f.write(f'\nlogzerr : {result_BAO_flat.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, H0 = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')
fig3 = corner.corner(result_BAO_flat.samples, weights=result_BAO_flat.weights, labels=[r'$\Omega_m$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999], bins=30, show_titles=True)
fig3.tight_layout()
fig3.savefig(Field_research_dir + '/Figs/nested_BAO_flat.png', dpi=300)

# 2-2. curved cosmology
print("Sampling BAO curved cosmology...")
result_BAO_curved = nestle.sample(un.Loglikelihood_BAO_curved, un.prior_BAO_curved, 3, method='single', npoints=1000, callback = nestle.print_progress) 
# saving the result
print("Saving BAO curved cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_BAO_curved_result.pkl', 'wb') as f:
    pickle.dump(result_BAO_curved, f)
p, cov = nestle.mean_and_cov(result_BAO_curved.samples, result_BAO_curved.weights)
with open(Field_research_dir + '/Results/Nested/nested_BAO_curved_summary.txt', 'w') as f:
    f.write(result_BAO_curved.summary())
    f.write(f'\nlogzerr : {result_BAO_curved.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, Omegalamb = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}, H0 = {p[2]:.3f} +/- {np.sqrt(cov[2,2]):.3f}')
fig4 = corner.corner(result_BAO_curved.samples, weights=result_BAO_curved.weights, labels=[r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999, 0.99999], bins=30, show_titles=True)
fig4.tight_layout()
fig4.savefig(Field_research_dir + '/Figs/nested_BAO_curved.png', dpi=300)


# 3. SN + BAO data
# 3-1. flat cosmology
print("Sampling SN + BAO flat cosmology...")
result_SN_BAO_flat = nestle.sample(un.Loglikelihood_SN_BAO_flat, un.prior_SN_BAO_flat, 2, method='single', npoints=200, callback = nestle.print_progress)
# saving the result
print("Saving SN + BAO flat cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_SN_BAO_flat_result.pkl', 'wb') as f:
    pickle.dump(result_SN_BAO_flat, f)
p, cov = nestle.mean_and_cov(result_SN_BAO_flat.samples, result_SN_BAO_flat.weights)
with open(Field_research_dir + '/Results/Nested/nested_SN_BAO_flat_summary.txt', 'w') as f:
    f.write(result_SN_BAO_flat.summary())
    f.write(f'\nlogzerr : {result_SN_BAO_flat.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, H0 = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}')
fig5 = corner.corner(result_SN_BAO_flat.samples, weights=result_SN_BAO_flat.weights, labels=[r'$\Omega_m$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999], bins=30, show_titles=True)
_ = fig5.suptitle('SN + BAO flat LCDM cosmology nested sampling result', fontsize=16)
fig5.tight_layout()
fig5.savefig(Field_research_dir + '/Figs/nested_SNBAO_flat.png', dpi=300)

# 3-2. curved cosmology
print("Sampling SN + BAO curved cosmology...")
result_SN_BAO_curved = nestle.sample(un.Loglikelihood_SN_BAO_curved, un.prior_SN_BAO_curved, 3, method='single', npoints=1000, callback = nestle.print_progress)
# saving the result
print("Saving SN + BAO curved cosmology result...")
with open(Field_research_dir + '/Results/Nested/nested_SN_BAO_curved_result.pkl', 'wb') as f:
    pickle.dump(result_SN_BAO_curved, f)
p, cov = nestle.mean_and_cov(result_SN_BAO_curved.samples, result_SN_BAO_curved.weights)
with open(Field_research_dir + '/Results/Nested/nested_SN_BAO_curved_summary.txt', 'w') as f:
    f.write(result_SN_BAO_curved.summary())
    f.write(f'\nlogzerr : {result_SN_BAO_curved.logzerr}')
    f.write(f'\n Best-fit paramters : Omegam = {p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}, Omegalamb = {p[1]:.3f} +/- {np.sqrt(cov[1,1]):.3f}, H0 = {p[2]:.3f} +/- {np.sqrt(cov[2,2]):.3f}')
fig4 = corner.corner(result_SN_BAO_curved.samples, weights=result_SN_BAO_curved.weights, labels=[r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$'],
                    quantiles=[0.16, 0.5, 0.84], range = [0.99999, 0.99999, 0.99999], bins=30, show_titles=True)

fig4.tight_layout()
fig4.savefig(Field_research_dir + '/Figs/nested_SNBAO_curved.png', dpi=300)
