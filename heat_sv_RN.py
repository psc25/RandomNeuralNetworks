import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

t = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
sig2 = sigma**2 + 2.0*lam*t

J = 200000
val_split = 0.2
Jtrain = int((1-val_split)*J)
ind_train = np.arange(Jtrain)
ind_test = np.arange(Jtrain, J)

activation = np.tanh

mm = [10, 20, 30]
NN = [16, 32, 64, 128, 256, 512, 1024]

err = np.nan*np.ones([len(mm), len(NN), 2])
tms = np.nan*np.ones([len(mm), len(NN)])
u1p = np.nan*np.ones([len(mm), len(NN), 100])
for mi in range(len(mm)):
    m = mm[mi]
    N = NN[0]
    V = np.random.normal(size = [J, m], scale = 0.5)
    Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    
    # Initialize variables
    A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = NN[0]))
    B = np.reshape(scs.multivariate_t(loc = 0.0, shape = 1.0).rvs(size = NN[0]), (1, -1))
    R = activation(np.matmul(V, A) - B)
    W = scl.lstsq(R[ind_train], Y[ind_train], lapack_driver = 'gelsy')[0]
    Y_train_pred = np.matmul(R, W)
    
    for Ni in range(len(NN)):
        N = NN[Ni]
        
        b = time.time()
        A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N))
        B = np.reshape(scs.multivariate_t(loc = 0.0, shape = 1.0).rvs(size = N), (1, -1))
        hid = np.matmul(V, A) - B
        batch_norm1 = np.mean(hid[ind_train])
        batch_norm2 = np.std(hid[ind_train])
        
        R = activation((hid - batch_norm1)/batch_norm2)
        W = scl.lstsq(R[ind_train], Y[ind_train], lapack_driver = 'gelsy')[0]
        Y_pred = np.matmul(R, W)
        e = time.time()
        err[mi, Ni, 0] = np.sqrt(np.mean(np.square(Y[ind_train] - Y_pred[ind_train])))
        err[mi, Ni, 1] = np.sqrt(np.mean(np.square(Y[ind_test] - Y_pred[ind_test])))
        tms[mi, Ni] = e-b
        
        u = 0.4*np.ones([100, m])
        u[:, 0] = np.linspace(-1.0, 1.0, 100)
        hid = np.matmul(u, A) - B
        R_u1p = activation((hid - batch_norm1)/batch_norm2)
        u1p[mi, Ni] = np.matmul(R_u1p, W).flatten()
        
        print("RN learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
    np.savetxt("heat_data/heat_sv_RN_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("heat_data/heat_sv_RN_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("heat_data/heat_sv_RN_u1p_" + str(mm[mi]) + ".csv", u1p[mi])