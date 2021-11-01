import numpy as np
from scipy import linalg as LA
import math



## See definition of distance between two spaces
## Page 9 of this paper: https://arxiv.org/abs/1808.03889

def space_distance(A, B, type=2):
    '''
    Return the distance between the column spaces of any two matrix A and B
    Args:
     A: np.matrix
     B: np.matrix, same shape of A
     type: 2 -- L2 norm
          'F' -- F norm

    Return: space distance between A and Ahat
    '''
    if not A.shape == B.shape:
        raise ValueError(
            'space_distance(): Matrices do have the same shape!')

    M = A @ LA.inv(A.transpose()@A) @ A.transpose() - \
        B @ LA.inv(B.transpose()@B) @ B.transpose()

    distance = LA.norm(M, ord=type)

    return distance


def get_metrics(X, F, R, C, alpha, R_hat, H_R, V_R, C_hat, H_C, V_C, k1_hat, k2_hat):
  
    p1, p2, _ = X.shape
    k1, k2, _ = F.shape
  
    # auxillary matrices
    H_R_inv = LA.inv(H_R)
    H_C_inv = LA.inv(H_C)
    
    # space distance norms, as before
    #distR = space_distance(R_hat, R)
    #distC = space_distance(C_hat, C)
    distR = LA.norm(R_hat - R @ H_R,2)/np.sqrt(p1)
    distC = LA.norm(C_hat - C @ H_C,2)/np.sqrt(p2)
  
    # get ||F_hat - H_R.T * F * H_C||_2
    F_hat = Fhat(X, R_hat, C_hat) 
    HFH =  np.einsum('ijk, kx ->ijx', F.transpose(2,1,0), H_R_inv.transpose()).transpose(2,1,0)
    HFH = np.einsum('ijk, jx -> ixk', HFH, H_C_inv.transpose())
    diffF = F_hat - HFH
    distF = np.mean(diffF, axis=2)
    distF_norm = LA.norm(distF, ord=2)
    
    # Rhat - RH \in \RR^{p1 x k1}
    errorR = math.sqrt(p2*T)*(R_hat - R @ H_R)
    errR_row = errorR[0,:] # get arbitrary row of each for error distributions
    
    # Cov_R
    # SigmaR0, _ = estim_cov((p1,p2), (k1,k2), X, R_hat, C_hat, F_hat, V_R, V_C, 0, 0)
    SigmaR0, _, alpha_opt1, alpha_opt2, _, _ = cov_loading(X, R_hat, C_hat, F_hat, V_R, V_C, 0, 0, alpha)
    # SigmaR0, _, alpha_opt1, alpha_opt2, _, _ = cov_loading_HAC(X, R_hat, C_hat, F_hat, V_R, V_C, 0, 0, alpha)
    SigmaR0_flat = np.reshape(np.ndarray.flatten(SigmaR0), (1,k1**2)) # flatten cov matrices
    
    # print('optimal alpha = ', alpha_opt1, ' // ', alpha_opt2)
    
    # errors: 1 x (5 + k1 + k1**2 + 2 + k1)
    num_cols = 5 + k1 + k1**2 + 2 + k1
    res = np.zeros((1, num_cols))
    res[:,:5] = [distR, distC, distF_norm, k1_hat, k2_hat] # space distances, k_hats (5)
    res[:,5:(5+k1)] = errR_row # R_hat - R * H (k1 x 1)
    res[:,(5+k1):(5+k1+k1**2)] = SigmaR0_flat # (k1**2)
    res[:,(5+k1+k1**2):(5+k1+k1**2+2)] = [alpha_opt1, alpha_opt2]
    res[:,(5+k1+k1**2+2):] = LA.inv(LA.cholesky(SigmaR0))@errR_row  # SigmaR0**(-1/2)@(R_hat - R * H) of (k1 x 1)
  
    return res
