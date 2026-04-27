import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

t = 1.0
K1 = 100
tt = np.linspace(0.0, t, K1)
dt1 = t/K1

mm = [10, 20, 30]
JJ = np.power(2, np.arange(12, 19))

ep = 5000
batch_size = 1000
eval_every = 250
lr = 5e-5
val_split = 0.2
activation = tf.tanh
init = tf.keras.initializers.GlorotNormal()
print_details = False

err = np.nan*np.ones([len(mm), len(JJ), 2])
tms = np.nan*np.ones([len(mm), len(JJ)])
u1p = np.nan*np.ones([len(mm), len(JJ), 100])
for mi in range(len(mm)):
    m = mm[mi]
    J = JJ[0]
    Jtrain = int((1-val_split)*J)
    ind_train = np.arange(Jtrain)
    ind_test = np.arange(Jtrain, J)
    nrbatch = int(np.ceil(Jtrain/batch_size))
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
    
    for Ji in range(len(JJ)):
        tf.reset_default_graph()
        J = JJ[Ji]
        Jtrain = int((1-val_split)*J)
        ind_train = np.arange(Jtrain)
        ind_test = np.arange(Jtrain, J)
        nrbatch = int(np.ceil(Jtrain/batch_size))
        N = 10*int(np.sqrt(J/np.log(J)))
        
        V = np.random.normal(size = [J, m], scale = 0.3)
        Y = np.exp(-0.5*np.sum(np.matmul(V-mut, Sigmat1)*(V-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.sqrt(np.linalg.det(Sigmat))
        
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
        tms[mi, Ji] = end1-begin1
        ind = np.nanargmin(res_loss[:, 1])
        err[mi, Ji] = res_loss[-1]
        u = 0.25*np.ones([100, m])
        u[:, 0] = np.linspace(-0.4, 0.4, 100)
        feed_dict = {inp: u}
        u1p[mi, Ji] = sess.run(out1, feed_dict).flatten()
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(np.arange(ep), res_loss[:, 0], "b-", label = "Train")
        plt.plot(np.arange(ep), res_loss[:, 1], "go", label = "Test")
        plt.legend()
        plt.ylim([0.0, 0.04])
        plt.savefig("fokker_data/fokker_sv_DN_learn_" + str(m) + "_" + str(J) + ".png", bbox_inches = 'tight', dpi = 500)
        plt.show()
        plt.close(fig)
        
        print("DN learned for m = " + str(m) + ", J = " + str(J) + ", N = " + str(N) + ", in " + '{:.4f}'.format(tms[mi, Ji]) + "s: in-sample " + '{:.4f}'.format(err[mi, Ji, 0]) + ", out-of-sample " + '{:.4f}'.format(err[mi, Ji, 1]))
        
        np.savetxt("fokker_data/fokker_sv_DN_err_" + str(mm[mi]) + ".csv", err[mi])
        np.savetxt("fokker_data/fokker_sv_DN_tms_" + str(mm[mi]) + ".csv", tms[mi])
        np.savetxt("fokker_data/fokker_sv_DN_u1p_" + str(mm[mi]) + ".csv", u1p[mi])