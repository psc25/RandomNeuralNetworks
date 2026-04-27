import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

t = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
sig2 = sigma**2 + 2.0*lam*t

val_split = 0.2

mm = [10, 20, 30]
JJ = np.power(2, np.arange(12, 19))

err = np.nan*np.ones([len(mm), len(JJ), 2])
tms = np.nan*np.ones([len(mm), len(JJ)])
u1p = np.nan*np.ones([len(mm), len(JJ), 100])
for mi in range(len(mm)):
    m = mm[mi]
    J = JJ[0]
    Jtrain = int((1-val_split)*J)
    ind_train = np.arange(Jtrain)
    ind_test = np.arange(Jtrain, J)
    N = 10*int(np.sqrt(J/np.log(J)))
    V = np.random.normal(size = [J, m], scale = 0.5)
    Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    
    # Initialize variables
    A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N))
    R1 = np.cos(np.matmul(V, A[:, :int(N/2)]))
    R2 = np.sin(np.matmul(V, A[:, int(N/2):]))
    R_train = np.concatenate([R1[ind_train], R2[ind_train]], axis = -1)
    W = scl.lstsq(R_train, Y[ind_train], lapack_driver = 'gelsy')[0]
    Y_train_pred = np.matmul(R_train, W)
    
    for Ji in range(len(JJ)):
        J = JJ[Ji]
        Jtrain = int((1-val_split)*J)
        ind_train = np.arange(Jtrain)
        ind_test = np.arange(Jtrain, J)
        N = 10*int(np.sqrt(J/np.log(J)))
        V = np.random.normal(size = [J, m], scale = 0.5)
        Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
        
        b = time.time()
        A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N))
        hid = np.matmul(V, A)
        batch_norm1 = np.mean(hid[ind_train])
        batch_norm2 = np.std(hid[ind_train])
        
        R1 = np.cos((hid[:, :int(N/2)] - batch_norm1)/batch_norm2)
        R2 = np.sin((hid[:, int(N/2):] - batch_norm1)/batch_norm2)
        R = np.concatenate([R1, R2], axis = -1)
        W = scl.lstsq(R[ind_train], Y[ind_train], lapack_driver = 'gelsy')[0]
        Y_pred = np.matmul(R, W)
        e = time.time()
        err[mi, Ji, 0] = np.sqrt(np.mean(np.square(Y[ind_train] - Y_pred[ind_train])))
        err[mi, Ji, 1] = np.sqrt(np.mean(np.square(Y[ind_test] - Y_pred[ind_test])))
        tms[mi, Ji] = e-b
        
        u = 0.4*np.ones([100, m])
        u[:, 0] = np.linspace(-1.0, 1.0, 100)
        hid = np.matmul(u, A)
        R_u1p1 = np.cos((hid[:, :int(N/2)] - batch_norm1)/batch_norm2)
        R_u1p2 = np.sin((hid[:, int(N/2):] - batch_norm1)/batch_norm2)
        R_u1p = np.concatenate([R_u1p1, R_u1p2], axis = -1)
        u1p[mi, Ji] = np.matmul(R_u1p, W).flatten()
        
        print("RT learned for m = " + str(m) + ", J = " + str(J) + ", N = " + str(N) + ", in " + '{:.4f}'.format(tms[mi, Ji]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ji, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ji, 1]))
        
    np.savetxt("heat_data/heat_sv_RT_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("heat_data/heat_sv_RT_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("heat_data/heat_sv_RT_u1p_" + str(mm[mi]) + ".csv", u1p[mi])