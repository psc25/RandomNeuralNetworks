import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

t = 1.0
K1 = 100
tt = np.linspace(0.0, t, K1)
dt1 = t/K1

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
    
    C1 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(m)
    C2 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(m)
    c2 = 0.1*np.ones([1, m], dtype = np.float32)/m
    D = 0.2*np.eye(m, dtype = np.float32)/np.sqrt(m)
    mu = np.zeros([K1+1, m], dtype = np.float32)
    Sigma = np.zeros([K1+1, m, m], dtype = np.float32)
    Sigma[0] = 0.5*np.eye(m, dtype = np.float32)/np.sqrt(m)
    for l in range(K1):
        mu[l+1] = mu[l] + (-np.matmul(C1, mu[l]) - c2)*dt1
        Sigma[l+1] = Sigma[l] + (-np.matmul(C1+C2, Sigma[l]) - np.matmul(Sigma[l], C1+C2) + 2*D)*dt1
        
    mut = mu[K1:]
    Sigmat = Sigma[K1]
    Sigmat1 = np.linalg.inv(Sigmat)
    
    V = np.random.normal(size = [J, m], scale = 0.3)
    Y = np.exp(-0.5*np.sum(np.matmul(V-mut, Sigmat1)*(V-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.sqrt(np.linalg.det(Sigmat))
    
    # Initialize variables
    A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N))
    R1 = np.cos(np.matmul(V, A[:, :int(N/2)]))
    R2 = np.sin(np.matmul(V, A[:, int(N/2):]))
    R = np.concatenate([R1, R2], axis = -1)
    W = scl.lstsq(R[ind_train], Y[ind_train], lapack_driver = 'gelsy')[0]
    Y_pred = np.matmul(R, W)
    
    for Ji in range(len(JJ)):
        J = JJ[Ji]
        Jtrain = int((1-val_split)*J)
        ind_train = np.arange(Jtrain)
        ind_test = np.arange(Jtrain, J)
        N = 10*int(np.sqrt(J/np.log(J)))
        
        V = np.random.normal(size = [J, m], scale = 0.3)
        Y = np.exp(-0.5*np.sum(np.matmul(V-mut, Sigmat1)*(V-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.sqrt(np.linalg.det(Sigmat))
        
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
        
        u = 0.25*np.ones([100, m])
        u[:, 0] = np.linspace(-0.4, 0.4, 100)
        hid = np.matmul(u, A)
        R_u1p1 = np.cos((hid[:, :int(N/2)] - batch_norm1)/batch_norm2)
        R_u1p2 = np.sin((hid[:, int(N/2):] - batch_norm1)/batch_norm2)
        R_u1p = np.concatenate([R_u1p1, R_u1p2], axis = -1)
        u1p[mi, Ji] = np.matmul(R_u1p, W).flatten()
        
        print("RT learned for m = " + str(m) + ", J = " + str(J) + ", N = " + str(N) + ", in " + '{:.4f}'.format(tms[mi, Ji]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ji, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ji, 1]))
        
    np.savetxt("fokker_data/fokker_sv_RT_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("fokker_data/fokker_sv_RT_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("fokker_data/fokker_sv_RT_u1p_" + str(mm[mi]) + ".csv", u1p[mi])