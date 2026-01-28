import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

ep = 10000
batch_size = 500
nrbatch = int(Jtrain/batch_size)
eval_every = 100
lr = 1e-5
activation = tf.tanh
init = tf.random_normal_initializer(stddev = 1e-5)
print_details = True

mm = [10, 20, 30]
NN = [16, 32, 64, 128, 256]

err = np.nan*np.ones([len(mm), len(NN), 2])
tms = np.nan*np.ones([len(mm), len(NN)])
u1p = np.nan*np.ones([len(mm), len(NN), K+1, 100])
for mi in range(len(mm)):
    m = mm[mi]
    N = NN[0]
    C1 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(np.float32(m))
    C2 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(np.float32(m))
    c2 = 0.1*np.ones([1, m], dtype = np.float32)/np.float32(m)
    D = 0.2*np.eye(m, dtype = np.float32)/np.sqrt(np.float32(m))
    mu = np.zeros([K1+1, m], dtype = np.float32)
    Sigma = np.zeros([K1+1, m, m], dtype = np.float32)
    Sigma[0] = 0.5*np.eye(m, dtype = np.float32)/np.sqrt(np.float32(m))
    for l in range(K1):
        mu[l+1] = mu[l] + (-np.matmul(C1, mu[l]) - c2)*dt1
        Sigma[l+1] = Sigma[l] + (-np.matmul(C1+C2, Sigma[l]) - np.matmul(Sigma[l], C1+C2) + 2*D)*dt1
        
    mut = np.expand_dims(mu[::int(K1/K)], axis = 1)
    Sigmat = Sigma[::int(K1/K)]
    Sigmat1 = np.linalg.inv(Sigmat)
    
    V = np.random.normal(size = [1, 1, J, m], scale = 0.3).astype(dtype = np.float32)
    Y = np.exp(-0.5*np.sum(np.matmul(V[0]-mut, Sigmat1)*(V[0]-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sigmat)), [-1, 1, 1])
    
    for Ni in range(len(NN)):
        tf.reset_default_graph()
        m = mm[mi]
        N = NN[Ni]
        
        begin1 = time.time()
        inp = tf.placeholder(shape = (1, 1, None, m), dtype = tf.float32)
        out = tf.placeholder(shape = (K+1, None, 1), dtype = tf.float32)
        
        mu_tf = tf.Variable(initial_value = init(shape = (N, 1, 1, m)), dtype = tf.float32)
        B0_tf = tf.Variable(initial_value = init(shape = (N, 1, m, m)), dtype = tf.float32)
        mask = tf.reshape(tf.linalg.band_part(tf.ones([m, m]), -1, 0), [1, 1, m, m])
        Si_tf = 6e-2*tf.reshape(tf.eye(m), [1, 1, m, m]) + tf.matmul(mask*B0_tf, mask*B0_tf, transpose_b = True)
        W = tf.Variable(initial_value = init(shape = (N, 1, 1, 1)), dtype = tf.float32)
        
        Si1_tf = tf.linalg.inv(Si_tf)
        R_train = tf.exp(-0.5*tf.reduce_sum(tf.matmul(inp-mu_tf, Si1_tf)*(inp-mu_tf), axis = -1, keepdims = True))/tf.pow(2.0*np.pi, m/2.0)/tf.reshape(tf.sqrt(tf.linalg.det(Si_tf)), [N, 1, 1, 1])
        out_train = tf.reduce_sum(W*R_train, axis = 0)
        loss_train = tf.reduce_mean(tf.square(out[0:1] - out_train))
        
        mu1 = [mu_tf[:, 0, 0]]
        Si1 = [Si_tf[:, 0]]
        for l in range(K1):
            mu1.append(mu1[-1] + (-tf.matmul(mu1[-1], C1.T) - c2)*dt1)
            Si1.append(Si1[-1] + (-tf.matmul(np.expand_dims(C1+C2, 0), Si1[-1]) - tf.matmul(Si1[-1], np.expand_dims(C1+C2, 0)) + 2*np.expand_dims(D, 0))*dt1)
        
        mut1 = tf.expand_dims(tf.stack(mu1, axis = 1)[:, ::int(K1/K)], 2)
        Sit1 = tf.stack(Si1, axis = 1)[:, ::int(K1/K)]
        Sit11 = tf.linalg.inv(Sit1)
        
        R_test = tf.exp(-0.5*tf.reduce_sum(tf.matmul(inp-mut1, Sit11)*(inp-mut), axis = -1, keepdims = True))/tf.pow(2.0*np.pi, m/2.0)/tf.reshape(tf.sqrt(tf.linalg.det(Sit1)), [N, K+1, 1, 1])
        out_test = tf.reduce_sum(W*R_test, axis = 0)
        loss_test = tf.reduce_mean(tf.square(out - out_test))
        
        global_step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        grads_and_vars = optimizer.compute_gradients(loss_train)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        res_loss = np.nan*np.ones([ep, 2])
        for i in range(ep):
            begin = time.time()
            loss1 = np.zeros([nrbatch, 1])
            np.random.shuffle(ind_train)
            for j in range(nrbatch):
                ind_batch = ind_train[(j*batch_size):((j+1)*batch_size)]
                feed_dict = {inp: V[:, :, ind_batch], out: Y[:, ind_batch]}
                _, loss1[j] = sess.run([train_op, loss_train], feed_dict)
                
            end = time.time()
            res_loss[i, 0] = np.sqrt(np.mean(loss1))
            if print_details:
                print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
                
            if (i+1) % eval_every == 0:
                begin = time.time()
                feed_dict = {inp: V[:, :, ind_test], out: Y[:, ind_test]}
                res_loss[i, 1] = np.sqrt(sess.run(loss_test, feed_dict))
                end = time.time()
                if print_details:
                    print("\nEvaluation on test data:")
                    print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
                    print("")
                    
        end1 = time.time()
        tms[mi, Ni] = end1-begin1
        ind = np.nanargmin(res_loss[:, 1])
        err[mi, Ni] = res_loss[-1]
        u = 0.25*np.ones([1, 1, 100, m])
        u1 = np.linspace(-0.4, 0.4, 100)
        u[0, 0, :, 0] = u1
        feed_dict = {inp: u}
        u1p[mi, Ni] = sess.run(out_test, feed_dict)[:, :, 0]
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(np.arange(ep), res_loss[:, 0], "b-", label = "Train")
        plt.plot(np.arange(ep), res_loss[:, 1], "go", label = "Test")
        plt.legend()
        plt.ylim([0.0, 0.07])
        plt.savefig("fokker_data/fokker_usv_DF_learn_" + str(m) + "_" + str(N) + ".png", bbox_inches = 'tight', dpi = 500)
        plt.show()
        plt.close(fig)
        
        print("DN learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
        np.savetxt("fokker_data/fokker_usv_DF_err_" + str(mm[mi]) + ".csv", err[mi])
        np.savetxt("fokker_data/fokker_usv_DF_tms_" + str(mm[mi]) + ".csv", tms[mi])
        np.savetxt("fokker_data/fokker_usv_DF_u1p_" + str(mm[mi]) + ".csv", np.reshape(u1p[mi], [len(NN), -1]))