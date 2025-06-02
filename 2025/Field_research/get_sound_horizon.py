#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Tools for predicting sound horizon for different cosmological parameters

Created for UST Field Research course, Cosmological Parameter Estimation
'''
import os.path
import camb
from camb import model, initialpower, CAMBdata
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

pars = camb.CAMBparams()
hubble = 20.
omega_m = 3.38
f_baryon =0.1543 
As = 2.4e-9
ns = 0.965
omega_cdm = 3.38*(1-f_baryon)
omega_baryon = omega_m*f_baryon
little_h = hubble/100.
pars.set_cosmology(TCMB=1.35,H0=hubble, ombh2=omega_baryon*little_h**2, omch2=omega_cdm*little_h**2)

pars.set_dark_energy() #re-set defaults
pars.InitPower.set_params(As=As,ns=ns)
#Not non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0.], kmax=100.0)
age = camb.get_age(pars)

print("Age of Universe (Gyr): {0:}".format(age))

results = camb.get_background(pars)
dict = results.get_derived_params()
print('r drag = {0:}'.format(dict['rdrag']))
print('z drag = {0:}'.format(dict['zdrag']))
