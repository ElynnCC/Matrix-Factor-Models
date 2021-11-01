import os
import numpy as np
import pickle
import math
from scipy import linalg as LA
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from collections import OrderedDict

## for UI
#from progressbar import ProgressBar

# for parallel computation
import multiprocessing.dummy as mp
from itertools import product, repeat

##
# self-defined functions
##
import MFM.gen_mat_obs as gen
import MFM.est_mfm as est
import MFM.metrics as per
import MFM.est_cov as aux
import MFM.util_mfm as util


def onemc(alpha, dim_obs, dim_latent, T, 
          setting, R, C, lam, mean_f_mat, 
          var_e, var_f, cov_e, cov_f, i, seed):

    print("Process MC #: " + str(i) + "\n")
    
    np.random.seed(seed)
    
    p, q = dim_obs
    k, r = dim_latent

    # Generate data
    X, F = gen.low_rank_matrix(setting, T, dim_obs, dim_latent, 
                               R, C, lam, mean_f_mat, var_e, var_f, cov_e, cov_f)
    
    
    # 
    X_tilde = est.matrix_trans(X, alpha)
    F_tilde = est.matrix_trans(F, alpha)
    
    # Estimate dimensions, loadings, and factors
    MR, MC = est.M_RC_hat(X_tilde)
    
    khat, QRhat, _, VRhat, _ = est.QVk_hat(MR, k)
    rhat, QChat, _, VChat, _ = est.QVk_hat(MC, r)
    
    Rhat = math.sqrt(p)*QRhat
    Chat = math.sqrt(q)*QChat
    
    Fhat = est.F_hat(X, Rhat, Chat)

    # no need for asymptotic normality
    # dim_latent_hat = (khat, rhat)

    # Estimation error
    spdistR = per.space_distance(Rhat, R)
    spdistC = per.space_distance(Chat, C)

    # \hat\bR - \bR\bH_R
    HR, HC = est.H(F_tilde, R, C, VRhat, VChat, Rhat, Chat)
    errorR = Rhat - R @ HR
    # errorC = Chat - C * HC  # did not include for saving time

    # get ||F_hat - H_R.T * F * H_C||_2
    # no need for asymptotic normality
    HR_inv = LA.inv(HR)
    HC_inv = LA.inv(HC)
    HFH = np.einsum('ijk, kx ->ijx', F.transpose(2, 1, 0),
                    HR_inv.transpose()).transpose(2, 1, 0)
    HFH = np.einsum('ijk, jx -> ixk', HFH, HC_inv.transpose())
    errorF = Fhat - HFH
    errorF_avg = np.mean(errorF, axis=2)
    errorF_avg_norm = LA.norm(errorF_avg, ord=2)

    row = 0
    col = 0
    # get arbitrary row of each for error distributions
    errorR_row = np.reshape(errorR[row, :], (1,k))
    Sigma_R_row, _, _, _, _, _ = aux.cov_loading_HAC(X, Rhat, Chat, Fhat, VRhat, VChat, row, col, alpha)
    #Sigma_R_row, _, _, _, _, _ = aux.cov_loading_HAC(X, Rhat, Chat, Fhat, VRhat, VChat, row, col, alpha)
    Sigma_R_row_flat = np.reshape(np.ndarray.flatten(Sigma_R_row), (1, k*k))  # flatten cov matrices

    temp1 = np.reshape([spdistR, spdistC, errorF_avg_norm, khat, rhat], (1,5))
    temp2 = np.hstack((temp1, errorR_row))
    res = np.hstack((temp2, Sigma_R_row_flat))

    return res



def model_serial(model_name, alpha, dim_obs, dim_latent, T, setting, deltas, lam, mean_f_mean, var_e, var_f, n_iter):
    p, q = dim_obs
    k, r = dim_latent
    deltaR, deltaC = deltas

    np.random.seed(6789)
    bdR = p**(-deltaR/2)
    bdC = q**(-deltaC/2)

    R = np.random.normal(1, 1, size=(p, k)) * bdR
    C = np.random.normal(1, 1, size=(q, r)) * bdC
    mean_f_mat = np.random.normal(mean_f_mean, 1, size=(k,r))

    res_model = np.zeros((n_iter, (k*k+k+5)))

    pbar = ProgressBar()
    for i_iter in pbar(range(n_iter)):
        res_model[i_iter, :] = onemc(alpha, dim_obs, dim_latent, T, setting, R, C, lam, mean_f_mat, var_e, var_f, cov_e, cov_f, seed=i_iter)

    return res_model



def model_parallel(n_cpu, model_name, alpha, dim_obs, dim_latent, T, setting, deltas, lam, mean_f_mean, var_e, var_f, cov_e, cov_f, n_iter):

    p, q = dim_obs
    k, r = dim_latent
    deltaR, deltaC = deltas

    np.random.seed()
    bdR = p**(-deltaR/2)
    bdC = q**(-deltaC/2)

    #R = np.matrix(np.random.uniform(low=-1, high=1, size=(p, k))) * bdR
    #C = np.matrix(np.random.uniform(low=-1, high=1, size=(q, r))) * bdC
    R = np.random.normal(1,1,size=(p,k)) * bdR
    C = np.random.normal(1,1,size=(q,r)) * bdC
    mean_f_mat = np.random.normal(mean_f_mean, 1, size=(k,r))

    pool = mp.Pool(n_cpu)
    single_params = [alpha, dim_obs, dim_latent, T, setting, R, C, lam, mean_f_mat, var_e, var_f, cov_e, cov_f]
    _SEED = 5555
    params = [single_params + [i, (i+1) * _SEED] for i in range(n_iter)]
    res_model = pool.starmap(onemc, params)
    pool.close()

    return res_model

##############################
##
##  Main starts here
##
##############################


RES_PATH = "results_asymp_normal"

# input
Roman = OrderedDict()
Roman[1] = "I"
Roman[2] = "II"
Roman[3] = "III"
Roman[4] = "IV"
setting = 4
model_name = Roman[setting]

n_iter = 500
n_cpu = 4

# pre-set
alpha_list = [-1, 0, 1]
X_dim_list = [(400, 400)]
F_dim = (3, 3)
Ta_list = [250]

dim_latent = (3, 3)
# mc simulation for all models
k, r = dim_latent

deltas = (0, 0)
lam = 1

mean_f_mean = 3

# VAR coefficients
var_e = 0
var_f = 0.9

cov_e = 0.1
cov_f = 1

# Estimate

spdist_df = pd.DataFrame(
    columns=['T', '(p, q)', '$\\alpha$', 'D($R$,$\hat{R}$)', 'D($C$,$\hat{C}$)'])
F_error_l2_df = pd.DataFrame(columns=['T', '(p, q)', '$\\alpha$', 'F_error'])
error_R_df = pd.DataFrame(
    columns=['T', '(p, q)', '$\\alpha$'] + list(range(k)))
error_R_std_df = pd.DataFrame(
    columns=['T', '(p, q)', '$\\alpha$'] + list(range(k)))

for alpha in alpha_list:
    for dim_obs in X_dim_list:
        for T in Ta_list:
            dim_obs_str = '-'.join(str(x) for x in dim_obs)
            prefix = '-'.join((model_name, dim_obs_str, str(T), str(mean_f_mean), str(cov_e), str(alpha)))

            print("Working on model " + prefix)
            
            res_model_list = model_parallel(n_cpu, model_name, alpha, dim_obs, dim_latent, T, setting, deltas, lam, mean_f_mean, var_e, var_f, cov_e, cov_f, n_iter)
            
            res_model = np.vstack(res_model_list)

            error_spdist, error_F, latent_dim_hat, error_R, cov_Rhat_flat = util.extract(
                res_model, k)

            cov_Rhat = np.reshape(cov_Rhat_flat, (k, k))
            sd_inv = LA.inv(LA.cholesky(cov_Rhat))
            error_R_std = error_R @ sd_inv.T

            spdist_new = util.frames_spdist(error_spdist, T, dim_obs, alpha)
            F_error_l2_new = util.frames_F_error(error_F, T, dim_obs, alpha)
            error_R_new = util.frames_R_rows(error_R, T, dim_obs, alpha)
            error_R_std_new = util.frames_R_rows(error_R_std, T, dim_obs, alpha)

            pickle.dump(spdist_new, open(os.path.join(RES_PATH, prefix+'-spdist.pkl'), 'wb'))
            pickle.dump(F_error_l2_new, open(os.path.join(RES_PATH, prefix+'-F-error.pkl'), 'wb'))
            pickle.dump(error_R_new, open(os.path.join(RES_PATH, prefix+'-R-error.pkl'), 'wb'))
            pickle.dump(error_R_std_new, open(os.path.join(RES_PATH, prefix+'-R-error-std.pkl'), 'wb'))

            spdist_df = spdist_df.append(spdist_new)
            F_error_l2_df = F_error_l2_df.append(F_error_l2_new)
            error_R_df = error_R_df.append(error_R_new)
            error_R_std_df = error_R_std_df.append(error_R_std_new)

pickle.dump(spdist_df, open(os.path.join(RES_PATH, 'spdist.pkl'), 'wb'))
pickle.dump(F_error_l2_df, open(os.path.join(RES_PATH, 'F-error.pkl'), 'wb'))
pickle.dump(error_R_df, open(os.path.join(RES_PATH, 'R-error.pkl'), 'wb'))
pickle.dump(error_R_std_df, open(os.path.join(RES_PATH, 'R-error-std.pkl'), 'wb'))

print("Done!")



