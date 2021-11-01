'''
Generate matrix obserations. 
'''

import os
import numpy as np
from scipy import linalg as LA
import pandas as pd

import statsmodels.api as sm

def gammas_small_od_1(p1, p2):
    # Error Cov where Gamma1 has 1/p_1 off diagonals,
    # Gamma2 is identity.
    od = 1/p1
    od2 = 1/p2
    Gamma1 = np.zeros((p1, p1))
    Gamma1.fill(od)
    Gamma1 += (1-od) * np.eye(p1)

    Gamma2 = np.zeros((p2, p2))
    Gamma2.fill(od2)
    Gamma2 += (1-od2) * np.eye(p2)

    return (Gamma1, Gamma2)


def VAR1(shape, size, coeff):
    nd1, nd2 = shape
    X = np.zeros((nd1, nd2, size))
    ma1 = np.array([1])
    ar1 = np.array([1, -coeff])
    for i in range(nd1):
        for j in range(nd2):
            ar1_process = sm.tsa.ArmaProcess(ar1, ma1)  # in 'statsmodels'
            X[i, j, :] = ar1_process.generate_sample(nsample=size)
    return X


def matrixvariate_normal(shape, size, mean=None, cov1=None, cov2=None):
    
    # current deal with iid latent matrix
    # return np.array

    if mean is None:
        mean = np.zeros(shape)
        
    if cov1 is None:
        cov1 = np.identity(shape[0])
        A = cov1
    else: 
        A = LA.cholesky(cov1)

    if cov2 is None:
        cov2 = np.identity(shape[1])
        B = cov2
    else:
        B = LA.cholesky(cov2)
        
    nd1, nd2 = shape
    
    # standard normal
    Z = np.random.normal(loc=0, scale=1, size=(nd1, nd2, size))
    
    # general normal
    X = np.zeros((nd1, nd2, size))
    for idx in range(size):
        X[:,:,idx] = mean + A@Z[:,:,idx]@B.T
    
    return X

def low_rank_matrix(setting,  # four settings see Chen and Fan (2019)
                    num_obs,
                    dim_obs,
                    dim_latent,
                    R, C,
                    lam=1,
                    mean_f_mat=None,
                    var_e=0, # for var model
                    var_f=0.9, # for var model
                    cov_e=0.1,
                    cov_f=1
                    ):  
    '''
    Generate samples from matrix factor models. 
    Out: (XX, FF, RR, CC)

    Todo:
    Change E here to heavy tail distributions
    - matrixvariate_t(...)
    - matrixvariate_lognormal(...)
    '''
    p1, p2 = dim_obs
    k1, k2 = dim_latent

    # setting is a global variable assigned in reporter
    
    if setting == 1:    # temporal cross-sectional uncorrelated
        E = matrixvariate_normal(shape=dim_obs, size=num_obs, mean=None, cov1=cov_e*np.eye(p1), cov2=cov_e*np.eye(p2))
        F = matrixvariate_normal(shape=dim_latent, size=num_obs)
    elif setting == 2:  # temporal correlated, cross-sectional uncorrelated
        E = VAR1(shape=dim_obs, size=num_obs, coeff=var_e)
        F = VAR1(shape=dim_latent, size=num_obs, coeff=var_f)
    elif setting == 3:  # temporal uncorrelated, cross-sectional correlated
        Ecov1, Ecov2 = gammas_small_od_1(p1, p2)
        E = matrixvariate_normal(shape=dim_obs, size=num_obs, cov1=Ecov1, cov2=Ecov2)
        F = matrixvariate_normal(shape=dim_latent, size=num_obs)
    elif setting == 4:  # non-zero mean, temporal cross-sectional uncorrelated
        E = matrixvariate_normal(shape=dim_obs, size=num_obs, mean=None, cov1=cov_e*np.eye(p1), cov2=cov_e*np.eye(p2))
        if mean_f_mat is None:
            mean_f_mat = 3*np.random.normal(loc=1,scale=1,size=dim_latent)
            print("Setting 4 but mean_f=None! Set mean_f randomly N(3, 1).")
        F = matrixvariate_normal(shape=dim_latent, size=num_obs, mean=mean_f_mat, cov1=cov_f*np.eye(k1), cov2=cov_f*np.eye(k2))  
        

    X = np.zeros((p1, p2, num_obs))
    for idx in range(num_obs):
        X[:, :, idx] = lam * R @ F[:, :, idx] @ C.T + E[:, :, idx]
    return (X, F)



def matrix_factor_model(dim_obs, dim_latent, T, lam, deltas, setting, errphi=1): 
    '''
    (M1) Data generator

    Return (X, F, R, C), the observed and latent factor time series.
    (??) should R and C be sampled inside of here or out?
       for now, they are sampled inside this function from
       uniform distributions depending on deltaR, deltaC.
       The setting determines how we calculate F and E.
    '''

    '''
    Change E here to heavy tail distributions
    - matrixvariate_t(...)
    - matrixvariate_lognormal(...)
    '''
    p1, p2 = dim_obs
    k1, k2 = dim_latent
    deltaR, deltaC = deltas

    E = np.zeros((p1, p2, T))
    F = np.zeros((k1, k2, T))
    
    # setting is a global variable assigned in reporter
    if setting == 1:
        E = matrixvariate_normal(shape=dim_obs, size=T)
        F = matrixvariate_normal(shape=dim_latent, size=T)
    elif setting == 2: 
        E = AR_one(shape=dim_obs, size=T, coeff=0.5)
        F = AR_one(shape=dim_latent, size=T, coeff=0.1)
    elif setting == 3:
        Ecov1, Ecov2 = gammas_small_od_1(p1,p2)
        E = matrixvariate_normal(shape=dim_obs, size=T, cov1=Ecov1, cov2=Ecov2)
        F = matrixvariate_normal(shape=dim_latent, size=T)
    elif setting == 5:
        Fcov1, Fcov2 = .5*np.identity(k1), .5*np.identity(k2) # small cov. F
        E = matrixvariate_normal(shape=dim_obs, size=T)
        mean_matrix = 10*np.random.normal(loc=1,scale=1,size=dim_latent) #np.ones(dim_latent)
        F = matrixvariate_normal(shape=dim_latent, size=T, mean=mean_matrix, cov1=Fcov1, cov2=Fcov2)
    elif setting == 6:
        Fcov1, Fcov2 = np.identity(k1), np.identity(k2) # small cov. F
        E = matrixvariate_normal(shape=dim_obs, size=T)
        mean_matrix = np.random.normal(loc=1,scale=1,size=dim_latent) #np.ones(dim_latent)
        F = matrixvariate_normal(shape=dim_latent, size=T, mean=mean_matrix, cov1=Fcov1, cov2=Fcov2)
    elif setting == 7:
        Fcov1, Fcov2 = np.identity(k1), np.identity(k2) # small cov. F
        E = matrixvariate_normal(shape=dim_obs, size=T)
        mean_matrix = 3*np.random.normal(loc=1,scale=1,size=dim_latent) #np.ones(dim_latent)
        F = matrixvariate_normal(shape=dim_latent, size=T, mean=mean_matrix, cov1=Fcov1, cov2=Fcov2)
    elif setting == 8:
        Fcov1, Fcov2 = np.identity(k1), np.identity(k2) # small cov. F
        E = matrixvariate_normal(shape=dim_obs, size=T)
        mean_matrix = 3*np.random.normal(loc=1,scale=1,size=dim_latent) #np.ones(dim_latent)
        F = matrixvariate_normal(shape=dim_latent, size=T, mean=mean_matrix, cov1=Fcov1, cov2=Fcov2)
    else:
        raise ValueError('Setting '+str(setting)+' is not defined yet.')
        
    bdR = p1**(-deltaR/2)
    bdC = p2**(-deltaC/2)

    #R = bdR * np.random.uniform(low=-1, high=1, size=(p1, k1))
    #C = bdC * np.random.uniform(low=-1, high=1, size=(p2, k2))
    R = bdR * np.random.normal(loc=1, scale=1, size=(p1, k1))
    C = bdC * np.random.normal(loc=1, scale=1, size=(p2, k2))

    X = np.zeros((p1, p2, T))
    for idx in range(T):
        X[:, :, idx] = lam * R @ F[:, :, idx] @ C.T + E[:, :, idx]
        
    return (X, F, R, C)


