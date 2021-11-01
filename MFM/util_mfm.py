
import numpy as np
from scipy import linalg as LA
from statsmodels.tsa.arima_process import ArmaProcess
import pickle
import pandas as pd


def simul_res_name(name, val):
    '''
    Func: 
        Generate result name of simulation.
    Input:
        val: value of model parameters (numbers)
        name: names of model parameters (string)
    Output:
        sss: a string of unique name
    '''
    
    name_val = ['-'.join(pair) for pair in zip(name,[str(a) for a in val])]
    
    return '_'.join(name_val)


def extract(res, k):   # contains all reps. in rows
    return (res[:, :2],  # spdistR, spdistC
            res[:, 2],  # F error
            res[:, 3:5],  # k, r
            res[:, 5:(5+k)],  # errorR_row at row = 0.
            np.mean(res[:, (5+k):], axis=0))  # cov_hatR, only mean is relevant


def save_frames(f1, f2, f3, f4, f5, fn1, fn2, fn3, fn4, fn5):
    pickle.dump(f1, open(fn1, 'wb'))
    pickle.dump(f2, open(fn2, 'wb'))
    pickle.dump(f3, open(fn3, 'wb'))
    pickle.dump(f4, open(fn4, 'wb'))
    pickle.dump(f5, open(fn5, 'wb'))


def frames_spdist(errors, T, dim_obs, alpha):
    '''
    Converts space distance error list into a pandas DataFrame for plotting purposes
    '''
    # retrieve the two space distance columns
    n_iter = errors.shape[0]
    p, q = dim_obs

    m = np.matrix(errors)[:, :2].tolist()
    x = ["("+str(p)+","+str(q)+")" for i in range(n_iter)]
    label = pd.DataFrame({'T': T, '(p, q)': x, '$\\alpha$': alpha})
    df = pd.DataFrame(m, columns=['D($R$,$\hat{R}$)', 'D($C$,$\hat{C}$)'])
    df = label.join(df)
    # dd = pd.melt(df, id_vars=['T', '(p, q)', '$\\alpha$'], value_vars=['D($R$,$\hat{R}$)', 'D($C$,$\hat{C}$)'], value_name="Space Distance", var_name = '')
    return df


def frames_F_error(errors, T, dim_obs, alpha):
    '''
    Converts space distance error list into a pandas DataFrame for plotting purposes
    '''
    m = errors.tolist()  # retrieve the column of errors
    # print(m)
    n_iter = errors.shape[0]
    p, q = dim_obs
    x = ["("+str(p)+","+str(q)+")" for i in range(n_iter)]
    label = pd.DataFrame({'T': T, '(p, q)': x, '$\\alpha$': alpha})
    df = pd.DataFrame(m, columns=['F_error'])
    df = label.join(df)
    # dd = pd.melt(df, id_vars=['T', '(p, q)', '$\\alpha$'], value_vars=['F_error'], value_name="L2-norm", var_name='')

    return df


def frames_R_rows(errors, T, dim_obs, alpha):
    '''
    Converts R error rows into a pandas DataFrame.
    '''
    # retrieve the two space distance columns
    n_iter = errors.shape[0]
    p, q = dim_obs

    m = errors.tolist()
    x = ["("+str(p)+","+str(q)+")" for i in range(n_iter)]
    label = pd.DataFrame({'T': T, '(p, q)': x, '$\\alpha$': alpha})
    df = pd.DataFrame(m)
    df = label.join(df)
    return df

# used in the application part


def varimax(x, normalize=True, eps=1e-5):
    p, nc = x.shape
    TT = np.eye(nc)
    d = 0
    # normalize
    sc = np.sqrt(np.matrix(np.sum(np.power(x, 2), axis=1)))
    div = np.tile(sc, (1, nc))
    x = np.divide(x, div)
    for i in range(1000):
        z = np.dot(x, TT)
        int1 = np.dot(np.matrix(np.ones(p)), np.power(z, 2))
        int1diag = np.diagflat(int1)
        int2 = np.dot(z, int1diag/p)
        int3 = np.power(z, 3) - int2
        B = np.dot(x.T, int3)
        u, s, vh = LA.svd(B)
        TT = np.dot(u, vh)
        dpast = d
        d = np.sum(s)
        if d < dpast * (1 + eps):
            break
    z = np.dot(x, TT)
    return np.multiply(z, div)
