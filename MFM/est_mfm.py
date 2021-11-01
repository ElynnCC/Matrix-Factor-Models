import numpy as np
import math
from scipy import linalg as LA

def matrix_trans(X, alpha):
    '''
        Func: calculate transform of matrix.
    '''
    T = X.shape[2]
    alpha_tilde = math.sqrt(alpha + 1) - 1
    X_avg = np.mean(X, axis=2)
    X_bar = np.repeat(X_avg[:,:,np.newaxis], T, axis=2)
    X_tilde = X + alpha_tilde*X_bar
    
    return X_tilde



def evecs_evals_decreasing(M, k):
    '''
    Func:
        Calculate top "k" real eigen vectors and eigen values of "M" in decreasing order. 
    In: 
        M: (p x p) np.array
        k: scalar
    Out:
        Vecs_d: p x k, eigenvectors
        Vals_d: k x k, diagonal, eigenvalues
    '''
    
    vals, Vecs = LA.eig(M) # the eigenvector with eigenvalue w[i] is U[:,i]

    # find largest k eigenvalues and ...
    # ... corresponding eigenvectors. 
    idx = np.argsort(vals.real)[::-1][:k]
    vals_d = vals.real[idx]
    Vecs_d = np.real(Vecs[:,idx])
    Vals_d = np.diagflat(vals_d)
    
    return (Vecs_d, Vals_d)

def XXtop(X):
    '''
    X X^T
    '''
    # 1/T X X^T using tensor dot.
    XXtop = np.tensordot(X, X, axes=([1, 2], [1, 2])) / np.prod(X.shape)
    return XXtop


# def M_hat(X, alpha=-1):
#     X_mean = np.mean(X, axis=2)
#     a2 = math.sqrt(alpha+1)-1
#     X_a = X + a2*X_mean[:, :, np.newaxis]
#     return (XXtop(X_a), XXtop(X_a.transpose(1, 0, 2)))


def M_RC_hat(X):
    
    X1, X2 = X, X.transpose(1, 0, 2)
    MR_hat = np.tensordot(X1, X2, axes=([1,2],[0,2]))/np.prod(X.shape)
    MC_hat = np.tensordot(X2, X1, axes=([1,2],[0,2]))/np.prod(X.shape)
    
    return (MR_hat, MC_hat)



##
# Wang et al. 2019
##


def Omega_hat(X, h):
    '''
    Returns Omega_hat as defined in Wang (2019), using tensordot.
    No missing values.
    '''
    # use np.tensordot() to avoid for loops as much as possible
    _, _, T = X.shape
    A = X[:, :, :T-h]
    B = X[:, :, h:]
    Omega = np.einsum('ijx,lkx->jkil', A, B)/(T-h)  # q x q x p x p

    return Omega


def M_hat_wang(X, h0=1):
    '''
    Returns M_hat as defined in Wang (2019).
    No missing values.
    '''
    p, _, _ = X.shape
    M_hat = np.zeros((p, p))
    for h in range(h0):
        Omega_h = Omega_hat(X, h+1)  # h from 1 to h0
        M_h = np.tensordot(Omega_h, Omega_h, axes=([0, 1, 3], [0, 1, 3]))
        M_hat += M_h

    return M_hat



def estim_k_simul(M_hat):
    '''
    Estimates k by the eigenvalue ratio test
    '''
    w, U = LA.eig(M_hat) #the eigenvector with eigenvalue w[i] is U[:,i]
    eigv = w.real[np.argsort(w.real)[::-1]] #sort largest to smallest
    end = math.floor(len(eigv)/2)
    ratios = eigv[1:end]/eigv[:end-1]
    k = np.argmin(ratios) + 1
    return k

def estimate_loading_matrices_chen(X, F, R, C, alpha, dim_obs, dim_latent):
    '''
    (M2) Solver
    Return (R_hat, C_hat) loading matrix estimates as in Chen (2019)
    '''
    p1, p2 = dim_obs
    k1, k2 = dim_latent
    
    X_tilde = matrix_trans(X, alpha)
    M_hatR, M_hatC = M_RC_hat(X_tilde)
  
    # get estimate for (k1,k2)
    k1_hat = estim_k_simul(M_hatR)
    k2_hat = estim_k_simul(M_hatC)
  
    R_hat, V_R = evecs_evals_decreasing(M_hatR, k1)
    C_hat, V_C = evecs_evals_decreasing(M_hatC, k2)
  
    # in the new paper, we assume 1/p1 R'R = I, 1/p2 C'C = I
    R_hat = math.sqrt(p1) * R_hat
    C_hat = math.sqrt(p2) * C_hat
    
    # calculate H according to (3.1) and (3.2) in the paper
    
    F_tilde = matrix_trans(F, alpha)
  
    H_R, H_C = H(F_tilde, R, C, V_R, V_C, R_hat, C_hat)
  
    return (R_hat, H_R, V_R, C_hat, H_C, V_C, k1_hat, k2_hat)


##
# Estimate Q and K from M
##


def QVk_hat(M_hat, k):
    '''
      Note here Q^\top Q = I
      To get Rhat and Chat defined in Chen & Fan 2019
      One need to multiply them by sqrt(p) and sqrt(q), respectively.
      This does not affect the space distance. 
    '''

    vals, vecs = LA.eig(M_hat)

    # rank eigvals & eigvecs in decreasing order
    vals = vals.real
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # estimate k by the eigval ratio test
    end = math.floor(len(vals)/2)
    ratios = vals[1:end]/vals[:end-1]
    k_hat = np.argmin(ratios) + 1

    # Q: eigenvecs
    Q_hat_k0 = vecs[:, :k]
    Q_hat_khat = vecs[:, :k_hat]

    # V: diag mat of eigenvals
    V_hat_k0 = np.diag(vals[:k])
    V_hat_khat = np.diag(vals[:k_hat])

    return (k_hat, Q_hat_k0, Q_hat_khat, V_hat_k0, V_hat_khat)

##
# Estimate F
##


def F_hat(X, Rhat, Chat):
    # F hat is estimated under the assumption that
    p1, p2, _ = X.shape
    Fhat = np.einsum('ik, ijt -> kjt', Rhat, X)
    Fhat = np.einsum('ijt, jx -> ixt', Fhat, Chat) / (p1*p2)
    return Fhat


##
# Estimate rotational matrix
##

def H(F, R, C, VRhat, VChat, Rhat, Chat):
    '''
    Returns H as in Chen (2019), F is the latent factors...with tensors
    '''
    T = F.shape[2]
    p1 = R.shape[0]
    p2 = C.shape[0]

    VRinv = LA.inv(VRhat)
    VCinv = LA.inv(VChat)

    FC = np.einsum('krt,qr->kqt', F, C)
    FCCF = np.tensordot(FC, FC, axes=([1, 2], [1, 2]))
    RRV = R.T@Rhat@VRinv
    HR = FCCF @ RRV/(p1*p2*T)

    FR = np.einsum('krt, pk->rpt', F, R)
    FRRF = np.tensordot(FR, FR, axes=([1, 2], [1, 2]))
    CCV = C.T@Chat@VCinv
    HC = FRRF @ CCV/(p1*p2*T)

    return (HR, HC)



##
# Oblique functions
##

# def H(F, R, C, V_R, V_C, R_hat, C_hat):
#     '''
#     Returns H as in Chen (2019), F is the latent factors...with tensors
#     '''
#     p1 = R.shape[0]
#     p2 = C.shape[0]
#     T = F.shape[2]
    
#     V_R_inv = LA.inv(V_R)
#     V_C_inv = LA.inv(V_C)

#     FC = np.einsum('ijk,jx->ixk',F,C.transpose())
    
#     FCCF = np.einsum('ijk,jyk->iyk',FC, FC.transpose((1,0,2)))
#     RRV = R.T@R_hat@V_R_inv
#     to_sum_R = np.einsum('ijk,jx->ixk',FCCF,RRV)
#     H_R = np.sum(to_sum_R, axis=2) / (p1*p2*T)

#     FR = np.einsum('ijk,jx->ixk',F.transpose((1,0,2)),R.transpose())
#     FRRF = np.einsum('ijk,jyk->iyk',FR, FR.transpose((1,0,2)))
#     CCV = C.transpose()@C_hat@V_C_inv
#     to_sum_C = np.einsum('ijk,jx->ixk', FRRF, CCV)
#     H_C = np.sum(to_sum_C, axis=2) / (p1*p2*T)

#     return (H_R, H_C)

# def H(F, R, C, V_R, V_C, R_hat, C_hat):
#     '''
#     Returns H as in Chen (2019), F is the latent factors...with tensors
#     '''
#     p1 = R.shape[0]
#     p2 = C.shape[0]
#     T = F.shape[2]
    
#     #H_R = np.zeros(V_R.shape)
#     #H_C = np.zeros(V_C.shape)
#     V_R_inv = LA.inv(V_R)
#     V_C_inv = LA.inv(V_C)

#     FC = np.einsum('krt,ir->kit',F,C) # k1 x p2 x T
#     FCCF = np.matrix(np.tensordot(FC, FC, axes=([1, 2], [1, 2]))) # np.einsum('kit,kjt->ijt',FC,FC)
#     RRV = R.T@R_hat@V_R_inv
#     H_R = FCCF@RRV / np.prod((p1,p2,T))

#     FR = np.einsum('krt,ik->rit',F,R) # k2 x p1 x T
#     FRRF = np.matrix(np.tensordot(FR, FR, axes=([1, 2], [1, 2])))
#     CCV = C.T@C_hat@V_C_inv
#     H_C = FRRF@CCV / np.prod((p1,p2,T))

#     return (H_R, H_C)



# def M_hat_alpha(X, alpha):

#     m_p1, m_p2, m_T = X.shape
#     alpha_tilde = math.sqrt(alpha + 1)-1
#     X_avg = np.mean(X, axis=2)
#     X_bar = np.repeat(X_avg[:,:,np.newaxis], m_T, axis=2)
#     X_tilde = X + alpha_tilde*X_bar

#     X1, X2 = X_tilde, X_tilde.transpose(1, 0, 2)
#     M_hat = np.tensordot(X1, X2, axes=([1,2],[0,2]))/ (m_T*m_p1*m_p2)
    
#     return M_hat

# def M12_hat_alpha(X, alpha):

#     m_p1, m_p2, m_T = X.shape
#     alpha_tilde = math.sqrt(alpha + 1)-1
#     X_avg = np.mean(X, axis=2)
#     X_bar = np.repeat(X_avg[:,:,np.newaxis], m_T, axis=2)
#     X_tilde = X + alpha_tilde*X_bar

#     X1, X2 = X_tilde, X_tilde.transpose(1, 0, 2)
#     M1_hat = np.tensordot(X1, X2, axes=([1,2],[0,2]))/ (m_T*m_p1*m_p2)
#     M2_hat = 
#     return (M1_hat, M2_hat)
