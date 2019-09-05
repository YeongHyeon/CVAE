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

def dat2canva(data, scale=1):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = (tmp * scale)
    if(dc == 1):
        canvas_rgb = np.ones((dh*numd, dw*numd, 3)).astype(np.float32)
        canvas_rgb[:, :, 0] = canvas[:, :, 0]
        canvas_rgb[:, :, 1] = canvas[:, :, 0]
        canvas_rgb[:, :, 2] = canvas[:, :, 0]
        canvas = canvas_rgb

    return canvas

def save_img(input, restore, recon, scale=1, savename=""):

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.title("Input\n(x)")
    plt.imshow(dat2canva(data=input, scale=1))
    plt.subplot(132)
    plt.title("Restoration\n(x to x-hat)")
    plt.imshow(dat2canva(data=restore, scale=1))
    plt.subplot(133)
    plt.title("Reconstruction\n(z to x-hat)")
    plt.imshow(dat2canva(data=recon, scale=1))
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(z_mu, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    if(normalize): scale = dataset.max_val
    else: scale = 1

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="tr_resotring")
    make_dir(path="tr_sampling")
    make_dir(path="tr_latent")

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    x_tr, y_tr, terminator = dataset.next_train(batch_size) # Initial batch
    dataset.reset_idx()
    for epoch in range(epochs):

        x_dump = np.zeros((batch_size, neuralnet.height, neuralnet.width, neuralnet.channel)).astype(np.float32)
        z_sample = gaussian_sample(mean=0, sigma=1, batch_size=batch_size, z_dim=neuralnet.z_dim)
        x_restore1, x_sample1 = sess.run([neuralnet.x_hat, neuralnet.x_sample], \
            feed_dict={neuralnet.x:x_dump, neuralnet.z:z_sample})
        save_img(input=x_dump, restore=x_restore1, recon=x_sample1, scale=scale, savename=os.path.join("tr_sampling", "%08d.png" %(epoch)))

        x_restore2, x_sample2, z_mu = sess.run([neuralnet.x_hat, neuralnet.x_sample, neuralnet.z_mu], \
            feed_dict={neuralnet.x:x_tr, neuralnet.z:z_sample})
        save_img(input=x_tr, restore=x_restore2, recon=x_sample2, scale=scale, savename=os.path.join("tr_resotring", "%08d.png" %(epoch)))

        latent_plot(z_mu=z_mu, y=y_tr, n=dataset.num_class, savename=os.path.join("tr_latent", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.
            if(x_tr is None): break

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr}, options=run_options, run_metadata=run_metadata)
            restore, kld, elbo = sess.run([neuralnet.mean_restore, neuralnet.mean_kld, neuralnet.loss], \
                feed_dict={neuralnet.x:x_tr})
            summary_writer.add_summary(summaries, iteration)

            if(iteration % 100 == 0):
                print("Epoch [%d / %d] (%d iteration)  Restore:%.3f, KLD:%.3f, Total:%.3f" \
                    %(epoch, epochs, iteration, restore, kld, elbo))
            iteration += 1
            if(terminator): break

        print("")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    while(True):
        x_te, y_te, terminator = dataset.next_test(batch_size) # y_te does not used in this prj.
        if(terminator): break
