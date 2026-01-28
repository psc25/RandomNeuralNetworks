import numpy as np
import scipy.stats as scs
import scipy.linalg as scl
import time

T = 1.0
K = 20
K1 = 100
dt1 = T/K1

tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1, 1])

J = 20000
val_split = 0.2
Jtrain = int((1-val_split)*J)
ind_train = np.arange(Jtrain)
ind_test = np.arange(Jtrain, J)

mm = [10, 20, 30]
NN = [16, 32, 64, 128, 256]

err = np.nan*np.ones([len(mm), len(NN), 2])
tms = np.nan*np.ones([len(mm), len(NN)])
u1p = np.nan*np.ones([len(mm), len(NN), K+1, 100])
for mi in range(len(mm)):
    m = mm[mi]
    N = NN[0]
    C1 = 0.1*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    C2 = 0.1*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    c2 = 0.1*np.ones([1, m], dtype = np.float32)/m
    D = 0.2*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    mu = np.zeros([K1+1, m], dtype = np.float32)
    Sigma = np.zeros([K1+1, m, m], dtype = np.float32)
    Sigma[0] = 0.5*np.eye(m, dtype = np.float32)/np.sqrt(m)
    for l in range(K1):
        mu[l+1] = mu[l] + (-np.matmul(C1[0], mu[l]) - c2)*dt1
        Sigma[l+1] = Sigma[l] + (-np.matmul(C1[0]+C2[0], Sigma[l]) - np.matmul(Sigma[l], C1[0]+C2[0]) + 2*D[0])*dt1
        
    mut = np.expand_dims(mu[::int(K1/K)], axis = 1)
    Sigmat = Sigma[::int(K1/K)]
    Sigmat1 = np.linalg.inv(Sigmat)
    
    V = np.random.normal(size = [1, 1, J, m], scale = 0.3).astype(dtype = np.float32)
    Y = np.exp(-0.5*np.sum(np.matmul(V[0]-mut, Sigmat1)*(V[0]-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sigmat)), [-1, 1, 1])
    
    # Initialize variables
    mu0 = scs.multivariate_t.rvs(size = N, shape = np.eye(m))
    B0 = np.reshape(scs.multivariate_t.rvs(size = N, shape = np.eye(m**2)), [N, m, m])
    Si0 = 1e-4*np.expand_dims(np.eye(m), 0) + np.matmul(B0, np.transpose(B0, [0, 2, 1]))
    
    mu1 = np.zeros([N, K1+1, m])
    Si1 = np.zeros([N, K1+1, m, m])
    Si1[:, 0] = np.tile(np.expand_dims(Sigma[0], 0), [N, 1, 1])
    for l in range(K1):
        mu1[:, l+1] = mu1[:, l] + (-np.matmul(mu1[:, l], C1[0]) - c2)*dt1
        Si1[:, l+1] = Si1[:, l] + (-np.matmul(C1+C2, Si1[:, l]) - np.matmul(Si1[:, l], C1+C2) + 2*D)*dt1
        
    mut1 = np.expand_dims(mu1[:, ::int(K1/K)], axis = 2)
    Sit1 = Si1[:, ::int(K1/K)]
    Sit11 = np.linalg.inv(Sit1)
    
    R = np.exp(-0.5*np.sum(np.matmul(V-mut1, Sit11)*(V-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sit1)), [N, K+1, 1, 1])
    W = scl.lstsq(np.reshape(R[:, 0, ind_train], [N, -1]).T , Y[0, ind_train].flatten(), lapack_driver = 'gelsy')[0]
    Y_pred = np.sum(R*np.reshape(W, [-1, 1, 1, 1]), axis = 0)
    
    for Ni in range(len(NN)):
        m = mm[mi]
        N = NN[Ni]
        
        b = time.time()
        mu0 = scs.multivariate_t.rvs(size = N, shape = np.eye(m))
        B0 = np.reshape(scs.multivariate_t.rvs(size = N, shape = np.eye(m**2)), [N, m, m])
        Si0 = np.matmul(B0, np.transpose(B0, [0, 2, 1]))
        
        mu1 = np.zeros([N, K1+1, m])
        Si1 = np.zeros([N, K1+1, m, m])
        Si1[:, 0] = np.tile(np.expand_dims(Sigma[0], 0), [N, 1, 1])
        for l in range(K1):
            mu1[:, l+1] = mu1[:, l] + (-np.matmul(mu1[:, l], C1[0]) - c2)*dt1
            Si1[:, l+1] = Si1[:, l] + (-np.matmul(C1+C2, Si1[:, l]) - np.matmul(Si1[:, l], C1+C2) + 2*D)*dt1
            
        mut1 = np.expand_dims(mu1[:, ::int(K1/K)], axis = 2)
        Sit1 = Si1[:, ::int(K1/K)]
        Sit11 = np.linalg.inv(Sit1)
        
        R = np.exp(-0.5*np.sum(np.matmul(V-mut1, Sit11)*(V-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sit1)), [N, K+1, 1, 1])
        W = scl.lstsq(np.reshape(R[:, 0, ind_train], [N, -1]).T , Y[0, ind_train].flatten(), lapack_driver = 'gelsy')[0]
        Y_pred = np.sum(R*np.reshape(W, [-1, 1, 1, 1]), axis = 0)
        
        err[mi, Ni, 0] = np.sqrt(np.mean(np.square(Y[:, ind_train] - Y_pred[:, ind_train])))
        err[mi, Ni, 1] = np.sqrt(np.mean(np.square(Y[:, ind_test] - Y_pred[:, ind_test])))
        
        e = time.time()
        tms[mi, Ni] = e-b
        
        u = 0.25*np.ones([1, 1, 100, m])
        u1 = np.linspace(-0.4, 0.4, 100)
        u[0, 0, :, 0] = u1
        R = np.exp(-0.5*np.sum(np.matmul(u-mut1, Sit11)*(u-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sit1)), [N, K+1, 1, 1])
        u1p[mi, Ni] = np.sum(R*np.reshape(W, [-1, 1, 1, 1]), axis = 0)[:, :, 0]
        
        print("RN learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        y = np.exp(-0.5*np.sum(np.matmul(u[0]-mut, Sigmat1)*(u[0]-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sigmat)), [-1, 1, 1])
        uu_plot, tt_plot = np.meshgrid(u1, tt.flatten())
        ax.plot_surface(tt_plot, uu_plot, u1p[mi, Ni], cmap = plt.cm.coolwarm, alpha = 0.7, linewidth = 0, antialiased = False)
        ax.plot_wireframe(tt_plot, uu_plot, y[:, :, 0], rstride = 3, cstride = 4, color = "black", linewidth = 0.4)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$u$')
        ax.set_zlabel('$f(t,u)$')
        plt.show()
        
    np.savetxt("fokker_data/fokker_usv_RN_err_" + str(mm[mi]) + ".csv", err[mi])
    np.savetxt("fokker_data/fokker_usv_RN_tms_" + str(mm[mi]) + ".csv", tms[mi])
    np.savetxt("fokker_data/fokker_usv_RN_u1p_" + str(mm[mi]) + ".csv", np.reshape(u1p[mi], [len(NN), -1]))