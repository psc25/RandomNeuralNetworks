import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

T = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
K = 20

tt = np.reshape(np.linspace(0.0, T, K+1, dtype = np.float32), [-1, 1, 1])
sig2 = sigma**2 + 2.0*lam*tt

val_split = 0.2

mm = [10, 20, 30]
JJ = np.power(2, np.arange(12, 19))

err = np.nan*np.ones([len(mm), len(JJ), 2])
tms = np.nan*np.ones([len(mm), len(JJ)])
u1p = np.nan*np.ones([len(mm), len(JJ), K+1, 100])
for mi in range(len(mm)):
    m = mm[mi]
    J = JJ[0]
    Jtrain = int((1-val_split)*J)
    ind_train = np.arange(Jtrain)
    ind_test = np.arange(Jtrain, J)
    N = 10*int(np.sqrt(J/np.log(J)))
    V = np.random.normal(size = [1, J, m], scale = 0.5).astype(np.float32)
    Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    
    # Initialize variables
    A = np.expand_dims(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N).T, axis = 0)
    B = np.reshape(scs.multivariate_t(loc = 0.0, shape = 1.0).rvs(size = N), (1, 1, -1))
    R = np.exp(-lam*tt*np.sum(np.square(A), axis = 1, keepdims = True))*np.cos(np.matmul(V, A) - B)
    W = scl.lstsq(np.reshape(np.transpose(R[:, ind_train], [2, 0, 1]), [N, -1]).T , Y[:, ind_train].flatten(), lapack_driver = 'gelsy')[0]
    Y_pred = np.sum(R*np.reshape(W, [1, 1, -1]), axis = -1, keepdims = True)
    
    for Ji in range(len(JJ)):
        J = JJ[Ji]
        Jtrain = int((1-val_split)*J)
        ind_train = np.arange(Jtrain)
        ind_test = np.arange(Jtrain, J)
        N = 10*int(np.sqrt(J/np.log(J)))
        V = np.random.normal(size = [1, J, m], scale = 0.5).astype(np.float32)
        Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
        
        b = time.time()
        A = np.expand_dims(scs.multivariate_t(loc = np.zeros(m), shape = np.identity(m)).rvs(size = N).T, axis = 0).astype(np.float32)
        B = np.reshape(scs.multivariate_t(loc = 0.0, shape = 1.0).rvs(size = N), (1, 1, -1)).astype(np.float32)
        R = np.exp(-lam*tt*np.sum(np.square(A), axis = 1, keepdims = True))*np.cos(np.matmul(V, A) - B)
        W = scl.lstsq(np.reshape(R[0, ind_train].T, [N, -1]).T , Y[0, ind_train].flatten(), lapack_driver = 'gelsy')[0]
        Y_pred = np.sum(R*np.reshape(W, [1, 1, -1]), axis = -1, keepdims = True)
        
        err[mi, Ji, 0] = np.sqrt(np.mean(np.square(Y[:, ind_train] - Y_pred[:, ind_train])))
        err[mi, Ji, 1] = np.sqrt(np.mean(np.square(Y[:, ind_test] - Y_pred[:, ind_test])))
        
        e = time.time()
        tms[mi, Ji] = e-b
        
        u = 0.4*np.ones([1, 100, m])
        u1 = np.linspace(-1.0, 1.0, 100)
        u[0, :, 0] = u1
        R = np.exp(-lam*tt*np.sum(np.square(A), axis = 1))*np.cos(np.matmul(u, A) - B)
        u1p[mi, Ji] = np.matmul(R, W)
        
        print("RF learned for m = " + str(m) + ", J = " + str(J) + ", N = " + str(N) + ", in " + '{:.4f}'.format(tms[mi, Ji]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ji, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ji, 1]))
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(u), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
        uu_plot, tt_plot = np.meshgrid(u1, tt.flatten())
        ax.plot_surface(tt_plot, uu_plot, u1p[mi, Ji], cmap = plt.cm.coolwarm, alpha = 0.7, linewidth = 0, antialiased = False)
        ax.plot_wireframe(tt_plot, uu_plot, y[:, :, 0], rstride = 3, cstride = 4, color = "black", linewidth = 0.4)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$u$')
        ax.set_zlabel('$f(t,u)$')
        plt.show()
        
    np.savetxt("heat_data/heat_usv_RF_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("heat_data/heat_usv_RF_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("heat_data/heat_usv_RF_u1p_" + str(mm[mi]) + ".csv", np.reshape(u1p[mi], [len(JJ), -1]))