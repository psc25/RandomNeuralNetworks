import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

T = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
K = 20

tt = np.reshape(np.linspace(0.0, T, K+1, dtype = np.float32), [-1, 1, 1])
sig2 = sigma**2 + 2.0*lam*tt

mm = [10, 20, 30]
JJ = np.power(2, np.arange(12, 19))

ep = 5000
batch_size = 1000
eval_every = 250
lr = 1e-4
val_split = 0.2
activation = tf.tanh
init = tf.keras.initializers.GlorotNormal()
print_details = False

err = np.nan*np.ones([len(mm), len(JJ), 2])
tms = np.nan*np.ones([len(mm), len(JJ)])
u1p = np.nan*np.ones([len(mm), len(JJ), K+1, 100])
for mi in range(len(mm)):
    m = mm[mi]
    J = JJ[0]
    Jtrain = int((1-val_split)*J)
    ind_train = np.arange(Jtrain)
    ind_test = np.arange(Jtrain, J)
    nrbatch = int(np.ceil(Jtrain/batch_size))
    N = 10*int(np.sqrt(J/np.log(J)))
    V = np.random.normal(size = [1, J, m], scale = 0.5).astype(np.float32)
    Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    
    for Ji in range(len(JJ)):
        tf.reset_default_graph()
        J = JJ[Ji]
        Jtrain = int((1-val_split)*J)
        ind_train = np.arange(Jtrain)
        ind_test = np.arange(Jtrain, J)
        nrbatch = int(np.ceil(Jtrain/batch_size))
        N = 10*int(np.sqrt(J/np.log(J)))
        
        V = np.random.normal(size = [1, J, m], scale = 0.5).astype(np.float32)
        Y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(V), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
        
        
        begin1 = time.time()
        inp = tf.placeholder(shape = (1, None, m), dtype = tf.float32)
        out = tf.placeholder(shape = (K+1, None, 1), dtype = tf.float32)
        
        A = tf.Variable(initial_value = init(shape = (1, m, N)), dtype = tf.float32)
        B = tf.Variable(initial_value = init(shape = (1, 1, N)), dtype = tf.float32)
        W = tf.Variable(initial_value = init(shape = (1, N, 1)), dtype = tf.float32)
        R_train = tf.exp(-lam*tt[0:1]*tf.reduce_sum(tf.square(A), axis = 1, keepdims = True))*tf.cos(tf.matmul(inp, A) - B)
        out_train = tf.matmul(R_train, W)
        loss_train = tf.reduce_mean(tf.square(out[0:1] - out_train))
        
        R_test = tf.exp(-lam*tt*tf.reduce_sum(tf.square(A), axis = 1, keepdims = True))*tf.cos(tf.matmul(inp, A) - B)
        out_test = tf.matmul(R_test, W)
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
                feed_dict = {inp: V[:, ind_batch], out: Y[:, ind_batch]}
                _, loss1[j] = sess.run([train_op, loss_train], feed_dict)
                
            end = time.time()
            res_loss[i, 0] = np.sqrt(np.mean(loss1))
            if print_details:
                print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
                
            if (i+1) % eval_every == 0:
                begin = time.time()
                feed_dict = {inp: V[:, ind_test], out: Y[:, ind_test]}
                res_loss[i, 1] = np.sqrt(sess.run(loss_test, feed_dict))
                end = time.time()
                if print_details:
                    print("\nEvaluation on test data:")
                    print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
                    print("")
                    
        end1 = time.time()
        tms[mi, Ji] = end1-begin1
        ind = np.nanargmin(res_loss[:, 1])
        err[mi, Ji] = res_loss[-1]
        u = 0.4*np.ones([1, 100, m])
        u1 = np.linspace(-1.0, 1.0, 100)
        u[0, :, 0] = u1
        feed_dict = {inp: u}
        u1p[mi, Ji] = sess.run(out_test, feed_dict)[:, :, 0]
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(np.arange(ep), res_loss[:, 0], "b-", label = "Train")
        plt.plot(np.arange(ep), res_loss[:, 1], "go", label = "Test")
        plt.legend()
        plt.ylim([0.0, 1.0])
        plt.savefig("heat_data/heat_usv_DF_learn_" + str(m) + "_" + str(N) + ".png", bbox_inches = 'tight', dpi = 500)
        plt.show()
        plt.close(fig)
        
        print("DF learned for m = " + str(m) + ", J = " + str(J) + ", N = " + str(N) + ", in " + '{:.4f}'.format(tms[mi, Ji]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ji, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ji, 1]))
        
        np.savetxt("heat_data/heat_usv_DF_err_" + str(mm[mi]) + ".csv", err[mi])
        np.savetxt("heat_data/heat_usv_DF_tms_" + str(mm[mi]) + ".csv", tms[mi])
        np.savetxt("heat_data/heat_usv_DF_u1p_" + str(mm[mi]) + ".csv", np.reshape(u1p[mi], [len(JJ), -1]))