import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

t = 1.0
lam = 4.0
sig = np.sqrt(2.0*lam*t)

J = 200000
val_split = 0.2
Jtrain = int((1-val_split)*J)
ind_train = np.arange(Jtrain)
ind_test = np.arange(Jtrain, J)

activation = np.tanh

mm = [10, 20, 30]
NN = [10, 25, 50, 100, 200, 400]

err = np.nan*np.ones([len(mm), len(NN), 2])
tms = np.nan*np.ones([len(mm), len(NN)])
evl = np.zeros([len(mm), len(NN)], dtype = np.int64)
u1p = np.nan*np.ones([len(mm), len(NN), 500])
for mi in range(len(mm)):
    m = mm[mi]
    R = 4.0*np.power(m, 0.4)
    V = np.random.normal(size = [J, m], scale = 1.0)
    ncp = np.sum(np.square(V/sig), axis = -1, keepdims = True)
    Y = scs.ncx2.cdf(np.square(R/sig), df = m, nc = ncp)
    
    # Initialize variables
    A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = NN[0]))
    R_train1 = np.cos(np.matmul(V[ind_train], A))
    R_train2 = np.sin(np.matmul(V[ind_train], A))
    R_train = np.concatenate([R_train1, R_train2], axis = -1)
    W = scl.lstsq(R_train, Y[ind_train], lapack_driver = 'gelsy')[0]
    Y_train_pred = np.matmul(R_train, W)
    
    for Ni in range(len(NN)):
        N = NN[Ni]
        
        # Accounting (for above): Generate V (Jtrain units) and compute Y := f(1,V) (Jtrain units)
        evl[mi, Ni] = evl[mi, Ni] + Jtrain + Jtrain
        
        b = time.time()
        A = np.transpose(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N))
        hid = np.matmul(V[ind_train], A)
        batch_norm1 = np.mean(hid)
        batch_norm2 = np.std(hid)
        # Accounting: Generate A (N units) and compute hid (Jtrain units)
        evl[mi, Ni] = evl[mi, Ni] + N + Jtrain
        
        R_train1 = np.cos((hid - batch_norm1)/batch_norm2)
        R_train2 = np.sin((hid - batch_norm2)/batch_norm2)
        R_train = np.concatenate([R_train1, R_train2], axis = -1)
        W = scl.lstsq(R_train, Y[ind_train], lapack_driver = 'gelsy')[0]
        Y_pred = np.matmul(R_train, W)
        e = time.time()
        err[mi, Ni, 0] = np.sqrt(np.mean(np.square(Y[ind_train] - Y_pred)))
        # Accounting: Least squares (Jtrain*N^2/2 + N^3/6 + O(Jtrain*N) units, see [Bjoerck, 1996])
        evl[mi, Ni] = evl[mi, Ni] + int(Jtrain*N**2/2.0) + int(N**3/6.0)
        
        hid = np.matmul(V[ind_test], A)
        R_test1 = np.cos((hid - batch_norm1)/batch_norm2)
        R_test2 = np.sin((hid - batch_norm2)/batch_norm2)
        R_test = np.concatenate([R_test1, R_test2], axis = -1)
        Y_pred = np.matmul(R_test, W)
        
        tms[mi, Ni] = e-b
        err[mi, Ni, 1] = np.sqrt(np.mean(np.square(Y[ind_test] - Y_pred)))
        
        u = 0.5*np.ones([500, m])
        u[:, 0] = np.linspace(-4.0, 4.0, 500)
        hid = np.matmul(u, A)
        R_u1p1 = np.cos((hid - batch_norm1)/batch_norm2)
        R_u1p2 = np.sin((hid - batch_norm2)/batch_norm2)
        R_u1p = np.concatenate([R_u1p1, R_u1p2], axis = -1)
        u1p[mi, Ni] = np.matmul(R_u1p, W).flatten()
        
        print("RT learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
    np.savetxt("heat_data/heat_RT_rnd_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("heat_data/heat_RT_rnd_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("heat_data/heat_RT_rnd_evl_" + str(mm[mi]) + ".csv", evl[mi])
    np.savetxt("heat_data/heat_RT_rnd_u1p_" + str(mm[mi]) + ".csv", u1p[mi])