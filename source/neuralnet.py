import tensorflow as tf

class CVAE(object):

    def __init__(self, height, width, channel, z_dim, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim])

        self.y, self.z_mu, self.z_sigma = None, None, None # will be initialized by 'build_model'
        self.build_model(input=self.x, ksize=self.k_size)

        # self.restore_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.y), axis=(1, 2, 3))
        self.restore_error = tf.reduce_sum(tf.square(self.x-self.y), axis=(1, 2, 3))

        self.kl_divergence = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.math.log(1e-8 + tf.square(self.z_sigma)) - 1, axis=(1))



        self.mean_restore = tf.reduce_mean(self.restore_error)
        self.mean_kld = tf.reduce_mean(self.kl_divergence)
        self.ELBO = tf.reduce_mean(self.kl_divergence + self.restore_error) # Evidence LowerBOund
        self.loss = self.ELBO

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.leaning_rate).minimize(self.loss)

        tf.compat.v1.summary.scalar('restore_error', self.mean_restore)
        tf.compat.v1.summary.scalar('kl_divergence', self.mean_kld)
        tf.compat.v1.summary.scalar('total loss', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def build_model(self, input, ksize=3):

        with tf.compat.v1.variable_scope("encoder"):
            output_enc = self.encoder(input=input, ksize=ksize)

        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            self.y = self.decoder(input=self.sample, ksize=ksize)
        self.x_sample = self.decoder(input=self.z, ksize=ksize)

    def encoder(self, input, ksize=3):

        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            print("Encode-1")
            self.conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 1, 16], activation="relu", name="conv1_1")
            self.conv1_2 = self.conv2d(input=self.conv1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], activation="relu", name="conv1_2")
            self.maxp1 = self.maxpool(input=self.conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")

            print("Encode-2")
            self.conv2_1 = self.conv2d(input=self.maxp1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 32], activation="relu", name="conv2_1")
            self.conv2_2 = self.conv2d(input=self.conv2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], activation="relu", name="conv2_2")
            self.maxp2 = self.maxpool(input=self.conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")

            print("Encode-3")
            self.conv3_1 = self.conv2d(input=self.maxp2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 64], activation="relu", name="conv3_1")
            self.conv3_2 = self.conv2d(input=self.conv3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], activation="relu", name="conv3_2")
            self.maxp3 = self.maxpool(input=self.conv3_2, ksize=2, strides=2, padding='SAME', name="max_pool3")

            print("Dense (Fully-Connected)")
            [n, h, w, c] = self.maxp3.shape
            self.fulcon_in = tf.compat.v1.reshape(self.maxp3, shape=[-1, h*w*c], name="fulcon_in")
            self.fulcon1 = self.fully_connected(input=self.fulcon_in, num_inputs=int(self.fulcon_in.shape[1]), \
                num_outputs=256, activation="relu", name="fullcon1")
            self.z_mu = self.fully_connected(input=self.fulcon1, num_inputs=int(self.fulcon1.shape[1]), \
                num_outputs=self.z_dim, activation="None", name="z_mu")
            self.z_sigma = self.fully_connected(input=self.fulcon1, num_inputs=int(self.fulcon1.shape[1]), \
                num_outputs=self.z_dim, activation="None", name="z_sigma")

            self.sample = self.sample_z(mu=self.z_mu, sigma=self.z_sigma)

        return self.sample

    def decoder(self, input, ksize=3):

        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):

            print("Decode-Dense")
            [n, h, w, c] = self.maxp3.shape
            self.fulcon2 = self.fully_connected(input=input, num_inputs=int(self.sample.shape[1]), \
                num_outputs=256, activation="relu", name="fullcon2")
            self.fulcon3 = self.fully_connected(input=self.fulcon2, num_inputs=int(self.fulcon2.shape[1]), \
                num_outputs=int(self.fulcon_in.shape[1]), activation="relu", name="fullcon3")
            self.fulcon_out = tf.compat.v1.reshape(self.fulcon3, shape=[-1, h, w, c], name="fulcon_out")

            print("Decode-1")
            self.convt1_1 = self.conv2d_transpose(input=self.fulcon_out, stride=2, padding='SAME', \
                output_shape=tf.shape(self.conv3_2), filter_size=[ksize, ksize, 64, 64], \
                dilations=[1, 1, 1, 1], activation="relu", name="convt1_1")
            self.convt1_2 = self.conv2d(input=self.convt1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], activation="relu", name="convt1_2")

            print("Decode-2")
            self.convt2_1 = self.conv2d_transpose(input=self.convt1_2, stride=2, padding='SAME', \
                output_shape=tf.shape(self.conv2_2), filter_size=[ksize, ksize, 32, 64], \
                dilations=[1, 1, 1, 1], activation="relu", name="convt2_1")
            self.convt2_2 = self.conv2d(input=self.convt2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], activation="relu", name="convt2_2")

            print("Decode-3")
            self.convt3_1 = self.conv2d_transpose(input=self.convt2_2, stride=2, padding='SAME', \
                output_shape=tf.shape(self.conv1_2), filter_size=[ksize, ksize, 16, 32], \
                dilations=[1, 1, 1, 1], activation="relu", name="convt3_1")
            self.convt3_2 = self.conv2d(input=self.convt3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], activation="relu", name="convt3_2")
            self.convt3_3 = self.conv2d(input=self.convt3_2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 1], activation="None", name="convt3_3")

            self.convt3_3 = tf.clip_by_value(self.convt3_3, 1e-12, 1-1e-12)

        return self.convt3_3

    def sample_z(self, mu, sigma):
        epsilon = tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        sample = mu + (sigma * epsilon)
        return sample

    def he_std(self, in_dim): return tf.sqrt(2.0 / in_dim)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" in activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" in activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" in activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        else: out = input

        return out

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        weight = tf.Variable(tf.random.normal(filter_size, \
            stddev=self.he_std(in_dim=filter_size[-2])), name='%s_w' %(name))
        bias = tf.Variable(tf.random.normal([filter_size[-1]], \
            stddev=self.he_std(in_dim=filter_size[-2])), name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation="relu", name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # [n, h, w, c] = output_shape
        # h, w, c = int(h), int(w), int(c)
        # print(n, h, w, c)
        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        weight = tf.Variable(tf.random.normal(filter_size, \
            stddev=self.he_std(in_dim=filter_size[-1])), name='%s_w' %(name))
        bias = tf.Variable(tf.random.normal([filter_size[-2]], \
            stddev=self.he_std(in_dim=filter_size[-1])), name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation="relu", name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        weight = tf.Variable(tf.random.normal([num_inputs, num_outputs], \
            stddev=self.he_std(in_dim=num_inputs)), name='%s_w' %(name))
        bias = tf.Variable(tf.random.normal([num_outputs], \
            stddev=self.he_std(in_dim=num_inputs)), name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
