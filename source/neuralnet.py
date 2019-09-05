import tensorflow as tf

class CVAE(object):

    def __init__(self, height, width, channel, z_dim, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []

        self.x_hat, self.logit, self.z_mu, self.z_sigma, self.x_sample = \
            self.build_model(input=self.x, random_z=self.z, ksize=self.k_size)

        # self.restore_error = tf.reduce_sum(tf.square(self.logit - self.x), axis=(1, 2, 3))
        self.restore_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit, labels=self.x), axis=(1, 2, 3))
        self.kl_divergence = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.math.log(1e-12 + tf.square(self.z_sigma)) - 1, axis=(1))

        self.mean_restore = tf.reduce_mean(self.restore_error)
        self.mean_kld = tf.reduce_mean(self.kl_divergence)
        self.ELBO = tf.reduce_mean(self.restore_error + self.kl_divergence) # Evidence LowerBOund
        self.loss = self.ELBO

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        tf.compat.v1.summary.scalar('restore_error', self.mean_restore)
        tf.compat.v1.summary.scalar('kl_divergence', self.mean_kld)
        tf.compat.v1.summary.scalar('total loss', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def build_model(self, input, random_z, ksize=3):

        # with tf.compat.v1.variable_scope("encode_var"):
        with tf.name_scope('encoder') as scope_enc:
            z_enc, z_mu, z_sigma = self.encoder(input=input, ksize=ksize)

        # with tf.compat.v1.variable_scope("decode_var", reuse=tf.compat.v1.AUTO_REUSE):
        with tf.name_scope('decoder') as scope_enc:
            logit = self.decoder(input=z_enc, ksize=ksize)
            x_hat = tf.compat.v1.nn.sigmoid(logit)
            x_sample = tf.compat.v1.nn.sigmoid(self.decoder(input=random_z, ksize=ksize))

        return x_hat, logit, z_mu, z_sigma, x_sample

    def encoder(self, input, ksize=3):

        print("Encode-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="relu", name="conv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="relu", name="conv1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")
        self.conv_shapes.append(tf.shape(conv1_2))

        print("Encode-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="relu", name="conv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="relu", name="conv2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")
        self.conv_shapes.append(tf.shape(conv2_2))

        print("Encode-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="relu", name="conv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="relu", name="conv3_2")
        maxp3 = self.maxpool(input=conv3_2, ksize=2, strides=2, padding='SAME', name="max_pool3")
        self.conv_shapes.append(tf.shape(conv3_2))

        print("Dense (Fully-Connected)")
        self.fc_shapes.append(maxp3.shape)
        [n, h, w, c] = self.fc_shapes[0]
        fulcon_in = tf.compat.v1.reshape(maxp3, shape=[-1, h*w*c], name="fulcon_in")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=256, activation="relu", name="fullcon1")
        z_mu = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim, activation="None", name="z_mu")
        z_sigma = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim, activation="None", name="z_sigma")

        z_enc = self.sample_z(mu=z_mu, sigma=z_sigma) # reparameterization trick

        return z_enc, z_mu, z_sigma

    def decoder(self, input, ksize=3):

        print("Decode-Dense")
        [n, h, w, c] = self.fc_shapes[0]
        fulcon2 = self.fully_connected(input=input, num_inputs=int(self.z_dim), \
            num_outputs=256, activation="relu", name="fullcon2")
        fulcon3 = self.fully_connected(input=fulcon2, num_inputs=int(fulcon2.shape[1]), \
            num_outputs=int(h*w*c), activation="relu", name="fullcon3")
        fulcon_out = tf.compat.v1.reshape(fulcon3, shape=[-1, h, w, c], name="fulcon_out")

        print("Decode-1")
        convt1_1 = self.conv2d_transpose(input=fulcon_out, stride=2, padding='SAME', \
            output_shape=self.conv_shapes[-1], filter_size=[ksize, ksize, 64, 64], \
            dilations=[1, 1, 1, 1], activation="relu", name="convt1_1")
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="relu", name="convt1_2")

        print("Decode-2")
        convt2_1 = self.conv2d_transpose(input=convt1_2, stride=2, padding='SAME', \
            output_shape=self.conv_shapes[-2], filter_size=[ksize, ksize, 32, 64], \
            dilations=[1, 1, 1, 1], activation="relu", name="convt2_1")
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="relu", name="convt2_2")

        print("Decode-3")
        convt3_1 = self.conv2d_transpose(input=convt2_2, stride=2, padding='SAME', \
            output_shape=self.conv_shapes[-3], filter_size=[ksize, ksize, 16, 32], \
            dilations=[1, 1, 1, 1], activation="relu", name="convt3_1")
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="relu", name="convt3_2")
        convt3_3 = self.conv2d(input=convt3_2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 1], activation="None", name="convt3_3")

        return convt3_3

    def sample_z(self, mu, sigma):

        # default of tf.random.normal: mean=0.0, stddev=1.0
        epsilon = tf.random.normal(tf.shape(mu), dtype=tf.float32)
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
        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.Variable(tf.random.normal(filter_size, \
                stddev=self.he_std(in_dim=filter_size[-2])), name='%s_w' %(name))
            bias = tf.Variable(tf.random.normal([filter_size[-1]], \
                stddev=self.he_std(in_dim=filter_size[-2])), name='%s_b' %(name))

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

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
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # [n, h, w, c] = output_shape
        # h, w, c = int(h), int(w), int(c)
        # print(n, h, w, c)
        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.Variable(tf.random.normal(filter_size, \
                stddev=self.he_std(in_dim=filter_size[-1])), name='%s_w' %(name))
            bias = tf.Variable(tf.random.normal([filter_size[-2]], \
                stddev=self.he_std(in_dim=filter_size[-1])), name='%s_b' %(name))

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

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
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.Variable(tf.random.normal([num_inputs, num_outputs], \
                stddev=self.he_std(in_dim=num_inputs)), name='%s_w' %(name))
            bias = tf.Variable(tf.random.normal([num_outputs], \
                stddev=self.he_std(in_dim=num_inputs)), name='%s_b' %(name))

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
