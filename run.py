import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp

def main():

    neuralnet = nn.VAE()

    dataset = dman.DataSet()

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( \\
        allow_soft_placement=True, \\
        log_device_placement=True))
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    tfp.test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='-')
    parser.add_argument('--batch', type=int, default=32, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
