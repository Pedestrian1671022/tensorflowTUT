import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_pixels = 96
batch_size = 40
epochs = 400
train_size = 16412
train_tfrecord = "faces_train.tfrecord"

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, filename

def generator(x, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.dense(x, units=6 * 6 * 64)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, units=24 * 24 * 128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 24, 24, 128])
        x = tf.layers.conv2d_transpose(x, 64, 2, strides=2)
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2)
        x = tf.nn.sigmoid(x)
        return x

def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 32, 5, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 64, 5, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 128, 3, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 256, 3, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
    return x

noise = tf.placeholder(tf.float32, shape=[None, 200])
image = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])

noise_image = generator(noise)

output_dis_real = discriminator(image)
output_dis_fake = discriminator(noise_image, reuse=True)

gan_output = discriminator(noise_image, reuse=True)
gen_label = tf.placeholder(tf.int32, shape=[None])

dis_output = tf.concat([output_dis_real, output_dis_fake], axis=0)
dis_label = tf.placeholder(tf.int32, shape=[None])


with tf.variable_scope("loss"):
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gan_output, labels=gen_label))
    dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dis_output, labels=dis_label))

    
with tf.variable_scope("train"):
    lr = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)
    # update_learning_rate = tf.assign(lr, lr * 0.8)
    train_gen = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=gen_loss, var_list=tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    train_dis = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=dis_loss, var_list=tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dataset_train = tf.data.TFRecordDataset([train_tfrecord])
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(400).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)
    # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #     print(var.name)

    for epoch in range(epochs):
        for step in range(int(train_size/batch_size)):
            batch_x, _ = sess.run(next_element_train)
            batch_x = batch_x/255.

            z = np.random.uniform(-1., 1., size=[batch_size, 200])
            batch_dis_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            batch_gen_y = np.ones([batch_size])

            _, _, gl, dl = sess.run([train_gen, train_dis, gen_loss, dis_loss], feed_dict={image: batch_x, noise: z, dis_label: batch_dis_y, gen_label: batch_gen_y})
            print('epoch: %d, step: %d, Generator Loss: %f, Discriminator Loss: %f' % (epoch, step, gl, dl))

        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            z = np.random.uniform(-1., 1., size=[4, 200])
            g = sess.run(noise_image, feed_dict={noise: z})
            for j in range(4):
                img = g[j]
                a[j][i].imshow(img[:, :, ::-1])
        f.savefig("faces_result/faces_%d.png" % epoch)
        plt.close()