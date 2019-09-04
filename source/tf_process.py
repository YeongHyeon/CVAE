import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gaussian_sample(batch_size, z_dim, mean=0, sigma=1):

    return np.random.normal(loc=mean, scale=sigma, size=(batch_size, z_dim)).astype(np.float32)

def save_img(data, savename=""):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp

    if(dc == 1):
        canvas_rgb = np.ones((dh*numd, dw*numd, 3)).astype(np.float32)
        canvas_rgb[:, :, 0] = canvas[:, :, 0]
        canvas_rgb[:, :, 1] = canvas[:, :, 0]
        canvas_rgb[:, :, 2] = canvas[:, :, 0]
        canvas = canvas_rgb
    plt.imsave(savename, canvas)

def training(sess, saver, neuralnet, dataset, epochs, batch_size):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint')

    make_dir(path="training")

    start_time = time.time()
    iteration = 0

    for epoch in range(epochs):

        x_dump = np.zeros((batch_size, neuralnet.height, neuralnet.width, neuralnet.channel)).astype(np.float32)
        z_sample = gaussian_sample(mean=0, sigma=1, batch_size=batch_size, z_dim=neuralnet.z_dim)
        x_sample = sess.run(neuralnet.x_sample, \
            feed_dict={neuralnet.x:x_dump, neuralnet.z:z_sample})
        save_img(data=x_sample, savename=os.path.join("training", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr})
            restore, kld, elbo = sess.run([neuralnet.mean_restore, neuralnet.mean_kld, neuralnet.loss], \
                feed_dict={neuralnet.x:x_tr})
            summary_writer.add_summary(summaries, iteration)

            if(iteration % 10 == 0):
                print("Epoch [%d / %d] (%d iteration)  Restore:%.3f, KLD:%.3f, Total:%.3f" \
                    %(epoch, epochs, iteration, restore, kld, elbo))
            iteration += 1
            if(terminator): break

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    while(True):
        x_te, y_te, terminator = dataset.next_test(batch_size) # y_te does not used in this prj.
        if(terminator): break
