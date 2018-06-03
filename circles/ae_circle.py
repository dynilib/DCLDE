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
num_feature = 2048

def generate_batch():
    r = np.random.normal(4,2,batch_size)
    phi = np.tile(np.linspace(0,2*np.pi,num_feature),(batch_size,1,1))
    batch = np.transpose(r.reshape(-1,1,1) * np.concatenate((np.cos(phi),np.sin(phi)),1),(0,2,1))
    return batch


def generate_unit():
    r = np.linspace(0.5,4.5,batch_size)
    phi = np.tile(np.linspace(0,2*np.pi,num_feature),(batch_size,1,1))
    batch = np.transpose(r.reshape(-1,1,1) * np.concatenate((np.cos(phi),np.sin(phi)),1),(0,2,1))
    return batch


unit = generate_unit()

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('Inputs'):
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, num_feature, 2))

    with tf.name_scope('Variables'):
        conv1_w = tf.Variable(tf.truncated_normal(
            [1,2,1,2], stddev=0.1), name='conv1_weight')
        conv2_w = tf.Variable(tf.truncated_normal(
            [1,2,2,1], stddev=0.1), name='conv2_weight')
        d1_w = tf.Variable(tf.truncated_normal(
            [2 * num_feature, num_feature], stddev=0.1), name='dense1_weight')
        d1_b = tf.Variable(tf.truncated_normal(
            [num_feature], stddev=0.1),name='dense1_bias')
        d2_w = tf.Variable(tf.truncated_normal(
            [num_feature, num_feature//4], stddev=0.1),name='dense2_weight')
        d2_b = tf.Variable(tf.truncated_normal(
            [num_feature//4], stddev=0.1),name='dense2_bias')
        d3_w = tf.Variable(tf.truncated_normal(
            [ num_feature//4, num_feature//16], stddev=0.1),name='dense3_weight')
        d3_b = tf.Variable(tf.truncated_normal(
            [num_feature//16], stddev=0.1),name='dense3_bias')
        d4_w = tf.Variable(tf.truncated_normal(
            [num_feature//16,1], stddev=0.1),name='dense4_weight')
        d4_b = tf.Variable(tf.truncated_normal(
            [1], stddev=0.1),name='dense4_bias')

        ud1_w = tf.Variable(tf.truncated_normal(
            [2 * num_feature, num_feature], stddev=0.1), name='undense1_weight')
        ud1_b = tf.Variable(tf.truncated_normal(
            [2 * num_feature], stddev=0.1),name='undense1_bias')
        ud2_w = tf.Variable(tf.truncated_normal(
            [num_feature, num_feature//4], stddev=0.1),name='undense2_weight')
        ud2_b = tf.Variable(tf.truncated_normal(
            [num_feature], stddev=0.1),name='undense2_bias')
        ud3_w = tf.Variable(tf.truncated_normal(
            [ num_feature//4, num_feature//16], stddev=0.1),name='undense3_weight')
        ud3_b = tf.Variable(tf.truncated_normal(
            [num_feature//4], stddev=0.1),name='undense3_bias')
        ud4_w = tf.Variable(tf.truncated_normal(
            [num_feature//16,1], stddev=0.1),name='undense4_weight')
        ud4_b = tf.Variable(tf.truncated_normal(
            [1], stddev=0.1),name='undense4_bias')

    with tf.name_scope('model'):
        def enc(input):
            conv1 = tf.nn.conv2d(tf.expand_dims(input,-1), conv1_w, [1,1,1,1],'SAME')
            conv2 = tf.nn.conv2d(conv1, conv2_w, [1,1,1,1],'SAME')
            dense1 = tf.matmul(tf.reshape(conv2,[batch_size,-1]), d1_w) + d1_b
            dense2 = tf.matmul(dense1, d2_w) + d2_b
            dense3 = tf.matmul(dense2, d3_w) + d3_b
            dense4 = tf.matmul(dense3, d4_w) + d4_b
            return dense4

        def dec(emb):
            undense4 = tf.matmul(emb, ud4_w, transpose_b=True) + ud4_b
            undense3 = tf.matmul(undense4, ud3_w, transpose_b=True) + ud3_b
            undense2 = tf.matmul(undense3, ud2_w, transpose_b=True) + ud2_b
            undense1 = tf.matmul(undense2, ud1_w, transpose_b=True) + ud1_b
            return tf.reshape(undense1, [batch_size, num_feature, 2])

        def model(data):
            with tf.name_scope('encoder'):
                z = enc(data)
            with tf.name_scope('decoder'):
                pred = dec(z)
            return pred, z
        prediction, embeddings = model(tf_train_dataset)


    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(tf_train_dataset,prediction)

    with tf.name_scope('Optimizer'):
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.00005, global_step, 20000, 0.1, False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    merged = tf.summary.merge_all()
    # Gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph,config=config) as session:
        #train_writer = tf.summary.FileWriter('ae_circle', session.graph)
        tf.global_variables_initializer().run()
        print('Initialized')
        ml=0
        fig, (ax1, ax2) = plt.subplots(2)
        plt.show(block=False)
        for step in trange(num_step):
            batch_data = generate_batch()
            feed_dict = {tf_train_dataset: batch_data}
            _, l = session.run([optimizer,loss], feed_dict=feed_dict)
            ml+=l
            if (step%num_visual_step == num_visual_step -1):
                ml/=num_visual_step
                tqdm.write(f'Minibatch mean loss from step {step +1 -num_visual_step} to step {step}: total : {ml}')
                ml=0
                feed_dict = {tf_train_dataset: unit}
                out, radius = session.run([prediction,embeddings], feed_dict=feed_dict)
                ax1.clear()
                ax2.clear()
                ax1.scatter(out[0,:,0],out[0,:,1],marker='.')
                ax1.scatter(out[20,:,0],out[20,:,1],marker='.')
                ax1.scatter(out[40,:,0],out[40,:,1],marker='.')
                ax1.scatter(out[63,:,0],out[63,:,1],marker='.')
                ax2.plot(np.squeeze(radius))
                plt.draw()
                plt.show(block=False)
                plt.pause(0.1)
        plt.show(block=True)