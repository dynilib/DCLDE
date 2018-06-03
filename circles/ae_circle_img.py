"""Copyright (C) 2018  DYNI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""


from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


num_step = 20000
num_visual_step = 40
batch_size = 64
num_feature = 128
radial_norm = np.expand_dims(np.abs(np.arange(num_feature).reshape(-1,1)+1j*np.arange(num_feature)-(1+1j)*(num_feature/2-0.5)),-1)


def generate_batch():
    r = (num_feature-65)/2*np.random.random(batch_size).reshape(-1,1,1,1)+ 25
    batch = 10*np.exp(-2*np.abs(r - radial_norm))
    return batch

def generate_unit():
    r = (num_feature - 65) / 2 * np.linspace(0,1,batch_size).reshape(-1, 1, 1, 1) + 25
    batch = 10*np.exp(-2*np.abs(r - radial_norm))
    return batch



const = generate_unit()

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('Inputs'):
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, num_feature, num_feature, 1))

    with tf.name_scope('Variables'):
        conv1_w = tf.Variable(tf.truncated_normal(
            [4,4,1,2], stddev=0.1), name='conv1_weight')
        conv2_w = tf.Variable(tf.truncated_normal(
            [8,8,2,4], stddev=0.1), name='conv2_weight')
        conv3_w = tf.Variable(tf.truncated_normal(
            [16,16,4,8], stddev=0.1), name='conv3_weight')

        d1_w = tf.Variable(tf.truncated_normal(
            [512, 64], stddev=0.1), name='dense1_weight')
        d1_b = tf.Variable(tf.truncated_normal(
            [64], stddev=0.1),name='dense1_bias')
        d2_w = tf.Variable(tf.truncated_normal(
            [64, 1], stddev=0.1),name='dense2_weight')
        d2_b = tf.Variable(tf.truncated_normal(
            [1], stddev=0.1),name='dense2_bias')

        conv4_w = tf.Variable(tf.truncated_normal(
            [3,3,1,8], stddev=0.1), name='conv3_weight')
        conv5_w = tf.Variable(tf.truncated_normal(
            [1,1,8,1], stddev=0.1), name='conv4_weight')


    with tf.name_scope('model'):
        def enc(input):
            conv1 = tf.nn.softplus(tf.nn.conv2d(input, conv1_w, [1,1,1,1],'SAME'))
            conv2 = tf.nn.softplus(tf.nn.conv2d(conv1, conv2_w, [1,4,4,1],'SAME'))
            conv3 = tf.nn.conv2d(conv2, conv3_w, [1,4,4,1],'SAME')
            reshape = tf.reshape(conv3,[batch_size,-1])
            dense1 = tf.matmul(reshape,d1_w) +d1_b
            dense2 = tf.matmul(dense1, d2_w) +d2_b
            return dense2

        def pinv(A, reltol=1e-6):
            # Compute the SVD of the input matrix A
            s, u, v = tf.svd(A)

            # Invert s, clear entries lower than reltol*s[0].
            atol = tf.reduce_max(s) * reltol
            s = tf.boolean_mask(s, s > atol)
            s_inv = tf.diag(1. / s)

            # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
            return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

        def dec(emb):
            undense2 =tf.matmul(emb - d2_b,pinv(d2_w))
            undense1 = tf.matmul(undense2 - d1_b,pinv(d1_w))
            unshape = tf.reshape(undense1, [batch_size, 8, 8, 8])
            unconv3 =  tf.nn.softplus(tf.nn.conv2d_transpose(unshape, conv3_w, [batch_size, 32, 32, 4], [1, 4, 4, 1]))
            unconv2 = tf.nn.softplus(tf.nn.conv2d_transpose(unconv3, conv2_w, [batch_size, num_feature, num_feature, 2], [1, 4, 4, 1]))
            unconv1 = tf.nn.conv2d_transpose(unconv2, conv1_w, [batch_size, num_feature, num_feature, 1], [1, 1, 1, 1])
            conv4 = tf.nn.softplus(tf.nn.conv2d(unconv1, conv4_w, [1,1,1,1], 'SAME'))
            conv5 = tf.nn.conv2d(conv4, conv5_w, [1,1,1,1], 'SAME')
            return conv5

        def model(data):
            with tf.name_scope('encoder'):
                z = enc(data)
            with tf.name_scope('decoder'):
                pred = dec(z)
            return pred, z
        prediction, embeddings = model(tf_train_dataset)


    with tf.name_scope('loss'):
        loss_rec = tf.losses.mean_squared_error(tf_train_dataset,prediction)
        loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss = loss_rec + 0.0001*loss_reg

    with tf.name_scope('Optimizer'):
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.0005, global_step, 20000, 0.1, False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    merged = tf.summary.merge_all()
    # Gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph,config=config) as session:
        #train_writer = tf.summary.FileWriter('ae_circle', session.graph)
        tf.global_variables_initializer().run()
        print('Initialized')
        ml=mc=mg=0

        fig = plt.figure()
        ax11 = plt.subplot2grid((3, 4), (0, 0))
        ax12 = plt.subplot2grid((3, 4), (1, 0))
        ax21 = plt.subplot2grid((3, 4), (0, 1))
        ax22 = plt.subplot2grid((3, 4), (1, 1))
        ax31 = plt.subplot2grid((3, 4), (0, 2))
        ax32 = plt.subplot2grid((3, 4), (1, 2))
        ax41 = plt.subplot2grid((3, 4), (0, 3))
        ax42 = plt.subplot2grid((3, 4), (1, 3))
        ax5 = plt.subplot2grid((3, 4), (2, 0), 1, 4)

        plt.show(block=False)
        for step in trange(num_step):
            batch_data = generate_batch()
            feed_dict = {tf_train_dataset: batch_data}
            _, l, lc, lg = session.run([optimizer, loss, loss_rec, loss_reg], feed_dict=feed_dict)
            ml += l
            mc += lc
            mg += lg
            if (step%num_visual_step == num_visual_step -1):
                ml/=num_visual_step
                mc/=num_visual_step
                mg/=num_visual_step
                tqdm.write(f'Minibatch mean loss from step {step +1 -num_visual_step} to step {step}: total : {ml}, reconstruction {mc}, regularization {mg}')
                ml=mc=mg=0
                feed_dict = {tf_train_dataset: const}
                out, radius = session.run([prediction,embeddings], feed_dict=feed_dict)

                ax11.clear()
                ax12.clear()
                ax21.clear()
                ax22.clear()
                ax31.clear()
                ax32.clear()
                ax41.clear()
                ax42.clear()
                ax5.clear()
                ax11.imshow(np.squeeze(const[0]))
                ax12.imshow(np.squeeze(out[0]))
                ax21.imshow(np.squeeze(const[20]))
                ax22.imshow(np.squeeze(out[20]))
                ax31.imshow(np.squeeze(const[40]))
                ax32.imshow(np.squeeze(out[40]))
                ax41.imshow(np.squeeze(const[63]))
                ax42.imshow(np.squeeze(out[63]))
                ax5.plot(np.squeeze(radius), marker='x')

                plt.draw()
                plt.show(block=False)
                plt.pause(0.1)
        plt.show(block=True)