import os, inspect, time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def training(sess, saver, neuralnet, dataset, epochs, batch_size):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint')

    start_time = time.time()
    iteration = 0
    for epoch in range(epochs):
        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size)

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr})
            likeli, kld, elbo = sess.run([neuralnet.marginal_likelihood, neuralnet.kl_divergence, neuralnet.loss], \
                feed_dict={neuralnet.x:x_tr})
            summary_writer.add_summary(summaries, iteration)

            print("Epoch [%d / %d] (%d iteration)  %.3f, %.3f, %.3f" %(epoch, epochs, iteration, likeli, kld, elbo))
            iteration += 1
            if(terminator): break

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    while(True):
        x_te, y_te, terminator = dataset.next_test(batch_size)
        if(terminator): break
