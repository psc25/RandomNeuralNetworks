import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

ep = 3000
batch_size = 500
nrbatch = int(Jtrain/batch_size)
eval_every = 100
lr = 5e-5
activation = tf.tanh
init = tf.random_normal_initializer()
print_details = True

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
    
    for Ni in range(len(NN)):
        tf.reset_default_graph()
        N = NN[Ni]
        
        begin1 = time.time()
        inp = tf.placeholder(shape = (None, m), dtype = tf.float32)
        out = tf.placeholder(shape = (None, 1), dtype = tf.float32)
        
        A = tf.Variable(initial_value = init(shape = (m, N)), dtype = tf.float32)
        W = tf.Variable(initial_value = init(shape = (N, 1)), dtype = tf.float32)
        out1 = tf.matmul(tf.cos(tf.matmul(inp, A[:, :int(N/2)])), W[:int(N/2)]) + tf.matmul(tf.sin(tf.matmul(inp, A[:, int(N/2):])), W[int(N/2):])
        
        loss = tf.reduce_mean(tf.square(out1 - out))
        
        global_step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        grads_and_vars = optimizer.compute_gradients(loss)
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
                feed_dict = {inp: V[ind_batch], out: Y[ind_batch]}
                _, loss1[j] = sess.run([train_op, loss], feed_dict)
                
            end = time.time()
            res_loss[i, 0] = np.sqrt(np.mean(loss1))
            if print_details:
                print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
                
            if (i+1) % eval_every == 0:
                begin = time.time()
                feed_dict = {inp: V[ind_test], out: Y[ind_test]}
                res_loss[i, 1] = np.sqrt(sess.run(loss, feed_dict))
                end = time.time()
                if print_details:
                    print("\nEvaluation on test data:")
                    print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
                    print("")
                    
        end1 = time.time()
        tms[mi, Ni] = end1-begin1
        ind = np.nanargmin(res_loss[:, 1])
        err[mi, Ni] = res_loss[-1]
        u = 0.4*np.ones([100, m])
        u[:, 0] = np.linspace(-1.0, 1.0, 100)
        feed_dict = {inp: u}
        u1p[mi, Ni] = sess.run(out1, feed_dict).flatten()
        
        print("TM learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
        np.savetxt("heat_data/heat_sv_DT_err_" + str(mm[mi]) + ".csv", err[mi])
        np.savetxt("heat_data/heat_sv_DT_tms_" + str(mm[mi]) + ".csv", tms[mi])
        np.savetxt("heat_data/heat_sv_DT_u1p_" + str(mm[mi]) + ".csv", u1p[mi])