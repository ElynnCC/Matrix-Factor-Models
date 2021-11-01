import numpy as np
import math
from scipy import linalg as LA


##
# Estimate covariance matrices for the loading matrices
##

def Omega_v(R_hat, C_hat, E_hat, F_hat, v, i, j, alpha):
    '''
    Return: (D_R, D_C) as in Section 4 in Chen and Fan (2019)
            D_R in R^{kxk} and D_C in R^{rxr}
    '''
    p1, p2, T = E_hat.shape
    k1, k2, _ = F_hat.shape
    
    if k1 != R_hat.shape[1] or k2 != C_hat.shape[1]:
        raise ValueError("F_t and R/C not compactible!")
    
    Omega_v_R11 = np.zeros((k1, k1))
    Omega_v_R12 = np.zeros((k1, k2))
    Omega_v_R21 = np.zeros((k2, k1))
    Omega_v_R22 = np.zeros((k2, k2))

    Omega_v_C11 = np.zeros((k2, k2))
    Omega_v_C12 = np.zeros((k2, k1))
    Omega_v_C21 = np.zeros((k1, k2))
    Omega_v_C22 = np.zeros((k1, k1))

    for t in range(v, T):
        F_t = F_hat[:,:,t]
        F_tv = F_hat[:,:,t-v]
        E_t_i = E_hat[i,:,t,None]
        E_tv_i = E_hat[i,:,t-v,None]
        E_t_j = E_hat[:,j,t,None]
        E_tv_j = E_hat[:,j,t-v,None]

        temp_R = C_hat.T @ E_t_i @ E_tv_i.T @ C_hat
        temp_C = R_hat.T @ E_t_j @ E_tv_j.T @ R_hat
        
        Omega_v_R11 += F_t @ temp_R @ F_tv.T
        Omega_v_C11 += F_t.T @ temp_C @ F_tv

        Omega_v_R12 += F_t @ temp_R
        Omega_v_C12 += F_t.T @ temp_C
        
        Omega_v_R21 += temp_R @ F_t.T
        Omega_v_C21 += temp_C @ F_t

        Omega_v_R22 += temp_R
        Omega_v_C22 += temp_C
    
    return (Omega_v_R11/(p2*T), Omega_v_R12/(p2*T), Omega_v_R21/(p2*T), Omega_v_R22/(p2*T), Omega_v_C11/(p1*T), Omega_v_C12/(p1*T), Omega_v_C21/(p1*T), Omega_v_C22/(p1*T))





def Omega_v_R(C_hat, E_hat, F_hat, v, i, alpha):
    '''
    Return: (D_R, D_C) as in Section 4 in Chen and Fan (2019)
            D_R in R^{kxk} and D_C in R^{rxr}
    '''
    _, p2, T = E_hat.shape
    k1, k2, _ = F_hat.shape
    if k2 != C_hat.shape[1]:
        raise ValueError("F_t and R/C not compactible!")
        
    Omega_v_R11 = np.zeros((k1, k1))
    Omega_v_R12 = np.zeros((k1, k2))
    Omega_v_R21 = np.zeros((k2, k1))
    Omega_v_R22 = np.zeros((k2, k2))

    for t in range(v, T):
        F_t = F_hat[:,:,t]
        F_tv = F_hat[:,:,t-v]
        E_t_i = E_hat[i,:,t,None]
        E_tv_i = E_hat[i,:,t-v,None]

        temp = C_hat.T @ E_t_i @ E_tv_i.T @ C_hat
        Omega_v_R11 += F_t @ temp @ F_tv.T
        Omega_v_R12 += F_t @ temp
        Omega_v_R21 += temp @ F_t.T
        Omega_v_R22 += temp
    
    return (Omega_v_R11/(p2*T), Omega_v_R12/(p2*T), Omega_v_R21/(p2*T), Omega_v_R22/(p2*T))





def Omega_v_C(R_hat, E_hat, F_hat, v, j, alpha):
    '''
    Return: D_C as in Section 4 in Chen and Fan (2019)
            D_C in R^{rxr}
    '''
    p1, _, T = E_hat.shape
    k1, k2, _ = F_hat.shape
    
    if k1 != R_hat.shape[1]:
        raise ValueError("F_t and R/C not compactible!")

    Omega_v_C11 = np.zeros((k2, k2))
    Omega_v_C12 = np.zeros((k2, k1))
    Omega_v_C21 = np.zeros((k1, k2))
    Omega_v_C22 = np.zeros((k1, k1))

    for t in range(v, T):
        F_t = F_hat[:,:,t]
        F_tv = F_hat[:,:,t-v]
        E_t_j = E_hat[:,j,t,None]
        E_tv_j = E_hat[:,j,t-v,None]

        temp = R_hat.T @ E_t_j @ E_tv_j.T @ R_hat
        Omega_v_C11 += F_t.T @ temp @ F_tv
        Omega_v_C12 += F_t.T @ temp
        Omega_v_C21 += temp @ F_t
        Omega_v_C22 += temp
    
    return (Omega_v_C11/(p1*T), Omega_v_C12/(p1*T), Omega_v_C21/(p1*T), Omega_v_C22/(p1*T))





def cov_loading(X, R_hat, C_hat, F_hat, VR, VC, i, j, alpha):
    '''
     only use v=0, it is the sample covariance matrix, not HAC cov
    '''
    p, q, T = X.shape
    k, r, _ = F_hat.shape
    
    #E_hat = X - np.tensordot(C_hat, np.tensordot(R_hat, F_hat, axes=(1, 0)), axes=(1, 1)).transpose((1, 0, 2))
    
    E_hat = np.zeros((p, q, T))
    for t in range(T):
        E_hat[:,:,t] = X[:,:,t] - R_hat @ F_hat[:,:,t] @ C_hat.T
    
    # remove mean to calculate variance 
    F_hat0 = F_hat    
    F_mean = np.mean(F_hat0, axis=2)
    F_hat = F_hat0 - np.repeat(F_mean[:,:,np.newaxis], T, axis=2)
    
    # v = 0 
    Omega_v_R11, Omega_v_R12, Omega_v_R21, Omega_v_R22, Omega_v_C11, Omega_v_C12, Omega_v_C21, Omega_v_C22 = Omega_v(R_hat, C_hat, E_hat, F_hat, 0, i, j, alpha)
    
   # choice of best alpha
    alpha_opt1 = -0.5 * np.trace(F_mean@Omega_v_R22@F_mean.T)**(-1) * np.trace(Omega_v_R12@F_mean.T + F_mean@Omega_v_R21)
    alpha_opt2 = -0.5 * np.trace(F_mean.T@Omega_v_C22@F_mean, 2)**(-1) * np.trace(Omega_v_C12@F_mean + F_mean.T@Omega_v_C21)

    VRinv = LA.inv(VR)
    VCinv = LA.inv(VC)
    
    if alpha == 0:
        Sigma_Ri = VRinv @ Omega_v_R11 @ VRinv
        Sigma_Cj = VCinv @ Omega_v_C11 @ VCinv
    else: 
        Sigma_Ri = VRinv @ (Omega_v_R11 + alpha*Omega_v_R12@F_mean.T + alpha*F_mean@Omega_v_R21 + alpha**2*F_mean@Omega_v_R22@F_mean.T) @ VRinv
        Sigma_Cj = VCinv @ (Omega_v_C11 + alpha*Omega_v_C12@F_mean + alpha*F_mean.T@Omega_v_C21 + alpha**2*F_mean.T@Omega_v_C22@F_mean) @ VCinv
        
    CCR = VRinv @ Omega_v_R11 @ VRinv
    BBR = VRinv @ (Omega_v_R12@F_mean.T + F_mean@Omega_v_R21) @ VRinv
    AAR = VRinv @ F_mean @ Omega_v_R22 @ F_mean.T @ VRinv
    
    CCC = VCinv @ Omega_v_C11 @ VCinv
    BBC = VCinv @ (Omega_v_C12@F_mean + F_mean.T@Omega_v_C21) @ VCinv
    AAC = VCinv @ F_mean.T @ Omega_v_C22 @ F_mean @ VCinv
    
    return (Sigma_Ri, Sigma_Cj, alpha_opt1, alpha_opt2, (CCR,BBR,AAR), (CCC,BBC,AAC))





def cov_loading_HAC(X, R_hat, C_hat, F_hat, VR, VC, i, j, alpha):
    '''
     HAC cov in Eqn. (4.2) and (4.3) with mr = log(qT) and mc = log(pt) for R and C respectively. 
    '''
    
    p, q, T = X.shape
    k, r, _ = F_hat.shape
    
    E_hat = np.zeros((p, q, T))
    for t in range(T):
        E_hat[:,:,t] = X[:,:,t] - R_hat @ F_hat[:,:,t] @ C_hat.T
        
    F_mean = np.mean(F_hat, axis=2)
    
    # remove mean to calculate variance 
    F_hat0 = F_hat    
    F_mean = np.mean(F_hat0, axis=2)
    F_hat = F_hat0 - np.repeat(F_mean[:,:,np.newaxis], T, axis=2)
    
    # v = 0 
    Omega_v_R11, Omega_v_R12, Omega_v_R21, Omega_v_R22, Omega_v_C11, Omega_v_C12, Omega_v_C21, Omega_v_C22 = Omega_v(R_hat, C_hat, E_hat, F_hat, 0, i, j, alpha)

    # v = 1 ... m
    # to make sure m / (qT)**(1/4) goes to 0
    mr = int(math.log(q*T))
    mc = int(math.log(p*T))
    mm = min(mr, mc)
    for v in range(1, mm+1):
        temp_v_R11, temp_v_R12, temp_v_R21, temp_v_R22, temp_v_C11, temp_v_C12, temp_v_C21, temp_v_C22 = \
            Omega_v(R_hat, C_hat, E_hat, F_hat, v, i, j, alpha)
        
        Omega_v_R11 += (temp_v_R11 + temp_v_R11.T)*(1-(v/(1+mr)))
        Omega_v_R12 += (temp_v_R12 + temp_v_R21.T)*(1-(v/(1+mr)))
        Omega_v_R21 += (temp_v_R21 + temp_v_R12.T)*(1-(v/(1+mr)))
        Omega_v_R22 += (temp_v_R22 + temp_v_R22.T)*(1-(v/(1+mr)))
        
        Omega_v_C11 += (temp_v_C11 + temp_v_C11.T)*(1-(v/(1+mc)))
        Omega_v_C12 += (temp_v_C12 + temp_v_C21.T)*(1-(v/(1+mc)))
        Omega_v_C21 += (temp_v_C21 + temp_v_C12.T)*(1-(v/(1+mc)))
        Omega_v_C22 += (temp_v_C22 + temp_v_C22.T)*(1-(v/(1+mc)))
        
    if mr > mc:
        for v in range(mm+1, mr+1):
            temp_v_R11, temp_v_R12, temp_v_R21, temp_v_R22 = Omega_v_R(C_hat, E_hat, F_hat, v, i, alpha)
            
            Omega_v_R11 += (temp_v_R11 + temp_v_R11.T)*(1-(v/(1+mr)))
            Omega_v_R12 += (temp_v_R12 + temp_v_R21.T)*(1-(v/(1+mr)))
            Omega_v_R21 += (temp_v_R21 + temp_v_R12.T)*(1-(v/(1+mr)))
            Omega_v_R22 += (temp_v_R22 + temp_v_R22.T)*(1-(v/(1+mr)))
    elif mr < mc:
        for v in range(mm+1, mc+1):
            temp_v_C11, temp_v_C12, temp_v_C21, temp_v_C22 = Omega_v_C(R_hat, E_hat, F_hat, v, j, alpha)
            
            Omega_v_C11 += (temp_v_C11 + temp_v_C11.T)*(1-(v/(1+mc)))
            Omega_v_C12 += (temp_v_C12 + temp_v_C21.T)*(1-(v/(1+mc)))
            Omega_v_C21 += (temp_v_C21 + temp_v_C12.T)*(1-(v/(1+mc)))
            Omega_v_C22 += (temp_v_C22 + temp_v_C22.T)*(1-(v/(1+mc)))

    # choice of best alpha
    alpha_opt1 = -0.5 * np.trace(F_mean@Omega_v_R22@F_mean.T)**(-1) * np.trace(Omega_v_R12@F_mean.T + F_mean@Omega_v_R21)
    alpha_opt2 = -0.5 * np.trace(F_mean.T@Omega_v_C22@F_mean, 2)**(-1) * np.trace(Omega_v_C12@F_mean + F_mean.T@Omega_v_C21)
    
    
    VRinv = LA.inv(VR)
    VCinv = LA.inv(VC)
    
    if alpha == 0:
        Sigma_Ri = VRinv @ Omega_v_R11 @ VRinv
        Sigma_Cj = VCinv @ Omega_v_C11 @ VCinv
    else: 
        Sigma_Ri = VRinv @ (Omega_v_R11 + alpha*Omega_v_R12@F_mean.T + alpha*F_mean@Omega_v_R21 + alpha**2*F_mean@Omega_v_R22@F_mean.T) @ VRinv
        
        Sigma_Cj = VCinv @ (Omega_v_C11 + alpha*Omega_v_C12@F_mean + alpha*F_mean.T@Omega_v_C21 + alpha**2*F_mean.T@Omega_v_C22@F_mean) @ VCinv
       
    CCR = VRinv @ Omega_v_R11 @ VRinv
    BBR = VRinv @ (Omega_v_R12@F_mean.T + F_mean@Omega_v_R21) @ VRinv
    AAR = VRinv @ F_mean @ Omega_v_R22 @ F_mean.T @ VRinv
    
    CCC = VCinv @ Omega_v_C11 @ VCinv
    BBC = VCinv @ (Omega_v_C12@F_mean + F_mean.T@Omega_v_C21) @ VCinv
    AAC = VCinv @ F_mean.T @ Omega_v_C22 @ F_mean @ VCinv
    
    return (Sigma_Ri, Sigma_Cj, alpha_opt1, alpha_opt2, (CCR,BBR,AAR), (CCC,BBC,AAC))




# def D(Q_hat, E_hat, F_hat, v, i, which='R'):
#     '''
#     Return D, as in the upper block of Eqn. (4.1) and (4.2)
#     '''
    
#     p1, p2, _ = E_hat.shape
    
#     if which == 'R':
#         k = F_hat.shape[0] 
#         D = np.zeros((k,k))
        
#         for t in range(v, T):
#             F_t = F_hat[:,:,t]
#             F_t_v = F_hat[:,:,t-v]
#             E_t_i = E_hat[i,:,t,None]
#             E_t_v_i = E_hat[i,:,t-v,None]
#             D += F_t @ Q_hat.T @ E_t_i @ E_t_v_i.T @ Q_hat @ F_t_v.T
#         D = D/(p2*(T-v))
        
#     else: # which == 'C'
#         k = F_hat.shape[1] 
#         D = np.zeros((k,k))
        
#         for t in range(v, T):
#             F_t = F_hat[:,:,t]
#             F_t_v = F_hat[:,:,t-v]
#             E_t_i = E_hat[:,i,t,None]
#             E_t_v_i = E_hat[:,i,t-v,None]
#             D += F_t.T @ Q_hat.T @ E_t_i @ E_t_v_i.T @ Q_hat @ F_t_v
#         D = D/(p1*(T-v))
#     return D






# def estim_cov(dim_obs, dim_latent, X, R_hat, C_hat, F_hat, V_R, V_C, i, j):
#     '''
#     Func: 
    
#     Input:
    
#     Output:
    
#     '''
#     p1, p2 = dim_obs
#     k1, k2 = dim_latent
#     E_hat = np.zeros((p1, p2, T))
#     #E_hat_t = Y_t - R_hat* F_hat_t * C_hat^T
#     for t in range(T):
#         E_hat[:,:,t] = X[:,:,t] - R_hat @ F_hat[:,:,t] @ C_hat.T
  
#     m_R = int(math.log(p2*T))
#     m_C = int(math.log(p1*T))
  
#     V_R_inv = LA.inv(V_R)
#     V_C_inv = LA.inv(V_C)
  
#     #Sigma_R_i
#     sum_D_R = D(C_hat, E_hat, F_hat, 0, i, 'R')
#     for v in range(m_R):
#         D_v = D(C_hat, E_hat, F_hat, v, i, 'R')
#         sum_D_R += (D_v + D_v.T) * (1 - (v / (1 + m_R)) )
#     Sigma_R_i = V_R_inv @ sum_D_R @ V_R_inv
  
#     #Sigma_C_i
#     sum_D_C = D(R_hat, E_hat, F_hat, 0, j, 'C')
#     for v in range(m_C):
#         D_v = D(R_hat, E_hat, F_hat, v, j, 'C')
#         sum_D_C += (D_v + D_v.T) * (1 - (v / (1 + m_C)) )
#     Sigma_C_j = V_C_inv @ sum_D_C @ V_C_inv
  
#     return (Sigma_R_i, Sigma_C_j)






##
# Estimate rotational matrix
##


# def H(F, R, C, VRhat, VChat, Rhat, Chat, alpha):
#     '''
#     Returns H as in Chen (2019), F is the latent factors...with tensors
#     '''
#     T = F.shape[2]
#     p = R.shape[0]
#     q = C.shape[0]

#     VRinv = LA.inv(VRhat)
#     VCinv = LA.inv(VChat)
    
#     F_mean = np.mean(F, axis=2)
#     a2 = math.sqrt(alpha+1)-1
#     Fa = F + a2*F_mean[:, :, np.newaxis]

#     FC = np.einsum('krt,ir->kit', Fa, C)
#     FCCF = np.matrix(np.tensordot(FC, FC, axes=([1, 2], [1, 2])))
#     RRV = R.transpose()*Rhat*VRinv
#     HR = FCCF * RRV/(p*q*T)

#     FR = np.einsum('krt, pk->rpt', Fa, R)
#     FRRF = np.matrix(np.tensordot(FR, FR, axes=([1, 2], [1, 2])))
#     CCV = C.transpose()*Chat*VCinv
#     HC = FRRF * CCV/(p*q*T)

#     return (np.matrix(HR), np.matrix(HC))





# def cov_loading_slower(X, R_hat, C_hat, F_hat, VR, VC, i, j, alpha):
#     # same result as the previous one
    
#     p, q, T = X.shape
#     k, r, _ = F_hat.shape
    
#     E_hat = X - np.tensordot(C_hat, np.tensordot(R_hat, F_hat, axes=(1, 0)), axes=(1, 1)).transpose((1, 0, 2))

#     mr = int(math.log(q*T))
#     mc = int(math.log(p*T))

#     # Sigma_R_i
#     sum_Omega_vR, sum_Omega_vC = Omega_v(R_hat, C_hat, E_hat, F_hat, 0, i, j, alpha)

#     for v in range(1,mr+1):
#         Omega_vR = Omega_vRR(C_hat, E_hat, F_hat, v, i, alpha)
#         sum_Omega_vR += (Omega_vR + Omega_vR.T)*(1-(v/(1+mr)))
#     for v in range(1,mc+1):
#         Omega_vC = Omega_vCC(R_hat, E_hat, F_hat, v, j, alpha)
#         sum_Omega_vC += (Omega_vC + Omega_vC.T)*(1-(v/(1+mc)))

#     VRinv = np.matrix(LA.inv(VR))
#     VCinv = np.matrix(LA.inv(VC))
    
#     if alpha != 0:
#         F_mean = np.matrix(np.mean(F_hat, axis=2))
        
#         A = np.concatenate((np.eye(k), alpha*F_mean), axis=1)
#         B = np.concatenate((np.eye(r), alpha*F_mean.T), axis=1)
        
#         Sigma_Ri = VRinv * A * np.matrix(sum_Omega_vR) * A.T * VRinv
#         Sigma_Cj = VCinv * B * np.matrix(sum_Omega_vC) * B.T * VCinv
#     else:
#         Sigma_Ri = VRinv * np.matrix(sum_Omega_vR[:k,:k]) * VRinv
#         Sigma_Cj = VCinv * np.matrix(sum_Omega_vC[:r,:r]) * VCinv

#     return (Sigma_Ri, Sigma_Cj)