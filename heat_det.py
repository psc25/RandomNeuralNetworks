import numpy as np
import scipy.stats as scs
import tensorflow as tf
import time

t = 1.0
lam = 4.0
sig = np.sqrt(2.0*lam*t)

J = 200000
val_split = 0.2
Jtrain = int((1-val_split)*J)
ind_train = np.arange(Jtrain)
ind_test = np.arange(Jtrain, J)

ep = 3000
batch_size = 500
nrbatch = int(Jtrain/batch_size)
eval_every = 100
lr = 1e-5
activation = tf.tanh
init = tf.random_normal_initializer()
print_details = True

mm = [10, 20, 30]
NN = [10, 50, 100, 200]

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
    
    for Ni in range(len(NN)):
        tf.reset_default_graph()
        N = NN[Ni]
        
        # Accounting (for above): Generate V (Jtrain*m units) and compute Y := f(V) (apply Jtrain-times the function f(1,\cdot))
        evl[mi, Ni] = evl[mi, Ni] + Jtrain*m + Jtrain
        
        begin1 = time.time()
        inp = tf.placeholder(shape = (None, m), dtype = tf.float32)
        out = tf.placeholder(shape = (None, 1), dtype = tf.float32)
        
        A = tf.Variable(initial_value = init(shape = (m, N)), dtype = tf.float32)
        B = tf.Variable(initial_value = init(shape = (1, N)), dtype = tf.float32)
        W = tf.Variable(initial_value = init(shape = (N, 1)), dtype = tf.float32)
        out1 = tf.matmul(activation(tf.matmul(inp, A) - B), W)
        
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
                
            # Accounting: Compute "out1" (Jtrain*(2*m+2) + Jtrain*(2*N-1) units), compute "loss" (2*Jtrain-1 units), and apply gradients (Jtrain*N*(m+1))
            evl[mi, Ni] = evl[mi, Ni] + Jtrain*(3*m+3) + 2*Jtrain-1 + Jtrain*N*(m+1)
            
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
        u = 0.5*np.ones([500, m])
        u[:, 0] = np.linspace(-4.0, 4.0, 500)
        feed_dict = {inp: u}
        u1p[mi, Ni] = sess.run(out1, feed_dict).flatten()
        
        print("DNN learned for m = " + str(mm[mi]) + ", N = " + str(NN[Ni]) + ", in " + '{:.4f}'.format(tms[mi, Ni]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ni, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ni, 1]))
        
        np.savetxt("heat_data/heat_det_err_" + str(mm[mi]) + ".csv", err[mi])
        np.savetxt("heat_data/heat_det_tms_" + str(mm[mi]) + ".csv", tms[mi])
        np.savetxt("heat_data/heat_det_evl_" + str(mm[mi]) + ".csv", evl[mi])
        np.savetxt("heat_data/heat_det_u1p_" + str(mm[mi]) + ".csv", u1p[mi])