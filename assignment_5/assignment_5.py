#!/usr/bin/env python
# coding: utf-8

# # Guillaume Payeur (260929164)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 200
# plt.rcParams.update({"text.usetex": True})
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import sys, platform, os
# import matplotlib
# from matplotlib import pyplot as plt
import time
# import corner


# In[3]:


#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


# # Q1
#
# I begin by running the test script

# In[4]:


def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

def ndiff(fun,x):
    delta = 1e-8
    # Calculating derivative
    f1 = 1/(2*delta)*(fun(x+delta)-fun(x-delta))
    return f1

# Initial guess for parameters
m=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])

# Doing itterations of Newton's method
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
for i in range(2):
    # Computing residuals
    errs=0.5*(planck[:,2]+planck[:,3]);
    model=get_spectrum(m)
    model=model[:len(spec)]
    resid=spec-model

    # Computing gradient of A numerically
    def fun_H0(H0):
        pars = np.concatenate((np.array([H0]),m[1:6]))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_H0 = ndiff(fun_H0,m[0])

    def fun_Omega_b(Omega_b):
        pars = np.concatenate((m[0:1],np.array([Omega_b]),m[2:6]))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_Omega_b = ndiff(fun_Omega_b,m[1])

    def fun_Omega_c(Omega_c):
        pars = np.concatenate((m[0:2],np.array([Omega_c]),m[3:6]))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_Omega_c = ndiff(fun_Omega_c,m[2])

    def fun_tau(tau):
        pars = np.concatenate((m[0:3],np.array([tau]),m[4:6]))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_tau = ndiff(fun_tau,m[3])

    def fun_A_s(A_s):
        pars = np.concatenate((m[0:4],np.array([A_s]),m[5:6]))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_A_s = ndiff(fun_A_s,m[4])

    def fun_n_s(n_s):
        pars = np.concatenate((m[0:5],np.array([n_s])))
        model=get_spectrum(pars)[:len(spec)]
        return model
    A_n_s = ndiff(fun_n_s,m[5])

    A_m = np.zeros((model.shape[0],6))
    A_m[:,0] = A_H0
    A_m[:,1] = A_Omega_b
    A_m[:,2] = A_Omega_c
    A_m[:,3] = A_tau
    A_m[:,4] = A_A_s*0
    A_m[:,5] = A_n_s

    # Computing delta m
    delta_m = np.linalg.pinv(A_m.T@A_m)@A_m.T@resid
    m = m + delta_m

    A_m = np.zeros((model.shape[0],6))
    A_m[:,0] = A_H0
    A_m[:,1] = A_Omega_b
    A_m[:,2] = A_Omega_c
    A_m[:,3] = A_tau
    A_m[:,4] = A_A_s
    A_m[:,5] = A_n_s

    # Computing delta m
    delta_m = np.linalg.pinv(A_m.T@A_m)@A_m.T@resid
    m = m + delta_m

# Printing best fit parameters
print(m)


# Plotting the model and the data to make sure the parameters are good

# In[8]:


# Now we approximate the uncertainty on the fit parameters. We use the known errors on each data point of the power spectrum, and assume that all data points are uncorrelated.

# In[9]:


# Calculating uncertainty on best fit parameters
Ninv = np.eye(errs.shape[0])*(1/errs)
cov = np.linalg.inv(A_m.T@Ninv@A_m)

# Printing uncertainty on best fit parameters
e_m = np.sqrt(np.diag(cov))
print(e_m)


# So the parameters and uncertainties we got via Newton's method are
# \begin{align}
#     H_0 &= (6.93\pm0.01)\text{x}10^1\\
#     \Omega_bh^2 &=(2.262\pm0.002)\text{x}10^{-2}\\
#     \Omega_ch^2 &=(1.154\pm0.002)\text{x}10^{-1}\\
#     \tau &= (6.4\pm0.3)\text{x}10^{-2}\\
#     A_s &= (2.10\pm0.01)\text{x}10^{-11}\\
#     n_s &= (9.774\pm0.007)\text{x}10^{-5}
# \end{align}
# Now we just save it all to a file

# In[23]:

# # Q3
#
# Now we use an MCMC to estimate the parameters and uncertainties

# In[11]:


# Function to make an MCMC chain
def chi2(pars,x,y,errs):
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=y-model
    chisq=np.sum((resid/errs)**2)
    return chisq

def mcmc(pars,step_size,x,y,fun,noise,nstep=10):
    # Initial chi2
    chi_cur=fun(pars,x,y,noise)
    # Making array to hold chains and chi2
    npar=pars.shape[0]
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)

    for i in range(nstep):
        print(i)
        trial_pars=pars+step_size*np.random.randn(npar)
        trial_chisq=fun(trial_pars,x,y,noise)
        delta_chisq=trial_chisq-chi_cur
        accept_prob=np.exp(-0.5*delta_chisq)
        accept=np.random.rand(1)<accept_prob
        if accept:
            pars=trial_pars
            chi_cur=trial_chisq
        chain[i,:]=pars
        chivec[i]=chi_cur
    return chain,chivec


# In[12]:


pars = np.repeat(np.expand_dims(m,axis=0),16,axis=0)
step_size = np.repeat(np.expand_dims(e_m,axis=0),16,axis=0)
x = np.repeat(np.expand_dims(planck[:,0],axis=0),16,axis=0)
y = np.repeat(np.expand_dims(planck[:,1],axis=0),16,axis=0)
fun = np.repeat(np.expand_dims(chi2,axis=0),16,axis=0)
noise = np.repeat(np.expand_dims(0.5*(planck[:,2]+planck[:,3]),axis=0),16,axis=0)

import concurrent.futures as futures

def run_mcmc():
    chains = []
    chis = []
    with futures.ProcessPoolExecutor() as pool:
        for chain_and_chi in pool.map(mcmc,pars,step_size,x,y,fun,noise):
            chains.append(chain_and_chi[0])
            chis.append(chain_and_chi[1])
    return chains,chis

if __name__ == '__main__':
    chains,chis = run_mcmc()
    for i in range(16):
        np.save('chains/chain_{}.npy'.format(i),chains[i])
        np.save('chains/chi{}.npy'.format(i),chis[i])
