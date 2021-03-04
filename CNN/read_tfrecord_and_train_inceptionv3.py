import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

image_pixels = 299
batch_size = 20
classes = 5
epochs = 200
train_size = 3894
train_tfrecord = "flowers_train.tfrecord"

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    label = tf.cast(features["label"], tf.int64)

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, label, filename

def conv_batch_normalization(prev_layer, layer_depth, is_training):

    gamma = tf.Variable(tf.ones([layer_depth]), trainable=True)
    beta = tf.Variable(tf.zeros([layer_depth]), trainable=True)

    pop_mean = tf.Variable(tf.zeros([layer_depth]), trainable=False)
    pop_variance = tf.Variable(tf.ones([layer_depth]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(prev_layer, [0, 1, 2], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(prev_layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(prev_layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)

def full_connection_batch_normalization(prev_layer, layer_depth, is_training):

    gamma = tf.Variable(tf.ones([layer_depth]), trainable=True)
    beta = tf.Variable(tf.zeros([layer_depth]), trainable=True)

    pop_mean = tf.Variable(tf.zeros([layer_depth]), trainable=False)
    pop_variance = tf.Variable(tf.ones([layer_depth]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(prev_layer, [0], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(prev_layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(prev_layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)

with tf.name_scope("input"):
    x_ = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="x_input")
    y_ = tf.placeholder(tf.int64, [None], name="y_input")
    is_training = tf.placeholder(tf.bool, None, name="is_training")

with tf.name_scope("InceptionV3"):
    with tf.name_scope("Conv2d_1a_3x3"):
        W_con1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1), trainable=True, name="W_Conv2d_1a_3x3")
        b_con1 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True, name="b_Conv2d_1a_3x3")
        h_conv1 = tf.nn.bias_add(tf.nn.conv2d(x_, W_con1, strides=[1, 2, 2, 1], padding="VALID"), b_con1)
        h_conv1_bn = conv_batch_normalization(prev_layer=h_conv1, layer_depth=32, is_training=is_training)
        
    with tf.name_scope("Conv2d_2a_3x3"):
        W_con2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1), trainable=True, name="W_Conv2d_2a_3x3")
        b_con2 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True, name="b_Conv2d_2a_3x3")
        h_conv2 = tf.nn.bias_add(tf.nn.conv2d(h_conv1_bn, W_con2, strides=[1, 1, 1, 1], padding="VALID"), b_con2)
        h_conv2_bn = conv_batch_normalization(prev_layer=h_conv2, layer_depth=32, is_training=is_training)
        
    with tf.name_scope("Conv2d_2b_3x3"):
        W_con3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), trainable=True, name="W_Conv2d_2b_3x3")
        b_con3 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Conv2d_2b_3x3")
        h_conv3 = tf.nn.bias_add(tf.nn.conv2d(h_conv2_bn, W_con3, strides=[1, 1, 1, 1], padding="SAME"), b_con3)
        h_conv3_bn = conv_batch_normalization(prev_layer=h_conv3, layer_depth=64, is_training=is_training)
        
    with tf.name_scope("MaxPool_3a_3x3"):
        h_conv3_bn_mp = tf.nn.max_pool(h_conv3_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
        
    with tf.name_scope("Conv2d_3b_1x1"):
        W_con4 = tf.Variable(tf.truncated_normal([1, 1, 64, 80], stddev=0.1), trainable=True, name="W_Conv2d_3b_1x1")
        b_con4 = tf.Variable(tf.constant(0.0, shape=[80]), trainable=True, name="b_Conv2d_3b_1x1")
        h_conv4 = tf.nn.bias_add(tf.nn.conv2d(h_conv3_bn_mp, W_con4, strides=[1, 1, 1, 1], padding="VALID"), b_con4)
        h_conv4_bn = conv_batch_normalization(prev_layer=h_conv4, layer_depth=80, is_training=is_training)
        
    with tf.name_scope("Conv2d_4a_3x3"):
        W_con5 = tf.Variable(tf.truncated_normal([3, 3, 80, 192], stddev=0.1), trainable=True, name="W_Conv2d_4a_3x3")
        b_con5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Conv2d_4a_3x3")
        h_conv5 = tf.nn.bias_add(tf.nn.conv2d(h_conv4_bn, W_con5, strides=[1, 1, 1, 1], padding="VALID"), b_con5)
        h_conv5_bn = conv_batch_normalization(prev_layer=h_conv5, layer_depth=192, is_training=is_training)
        
    with tf.name_scope("MaxPool_5a_3x3"):
        h_conv5_bn_mp = tf.nn.max_pool(h_conv5_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
        
    with tf.name_scope("Mixed_5b"):
        W_con6_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 192, 64], stddev=0.1), trainable=True, name="W_Mixed_5b_b1_1")
        b_con6_b1_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5b_b1_1")
        h_conv6_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv5_bn_mp, W_con6_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_con6_b1_1)
        h_conv6_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv6_b1_1, layer_depth=64, is_training=is_training)

        W_con6_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 192, 48], stddev=0.1), trainable=True, name="W_Mixed_5b_b2_1")
        b_con6_b2_1 = tf.Variable(tf.constant(0.0, shape=[48]), trainable=True, name="b_Mixed_5b_b2_1")
        h_conv6_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv5_bn_mp, W_con6_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_con6_b2_1)
        h_conv6_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv6_b2_1, layer_depth=48, is_training=is_training)
        W_con6_b2_2 = tf.Variable(tf.truncated_normal([5, 5, 48, 64], stddev=0.1), trainable=True, name="W_Mixed_5b_b2_2")
        b_con6_b2_2 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5b_b2_2")
        h_conv6_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv6_bn_b2_1, W_con6_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_con6_b2_2)
        h_conv6_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv6_b2_2, layer_depth=64, is_training=is_training)

        W_con6_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 192, 64], stddev=0.1), trainable=True,name="W_Mixed_5b_b3_1")
        b_con6_b3_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5b_b3_1")
        h_conv6_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv5_bn_mp, W_con6_b3_1, strides=[1, 1, 1, 1], padding="SAME"),b_con6_b3_1)
        h_conv6_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv6_b3_1, layer_depth=64, is_training=is_training)
        W_con6_b3_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1), trainable=True,name="W_Mixed_5b_b3_2")
        b_con6_b3_2 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5b_b3_2")
        h_conv6_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv6_bn_b3_1, W_con6_b3_2, strides=[1, 1, 1, 1], padding="SAME"),b_con6_b3_2)
        h_conv6_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv6_b3_2, layer_depth=96, is_training=is_training)
        W_con6_b3_3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1), trainable=True,name="W_Mixed_5b_b3_3")
        b_con6_b3_3 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5b_b3_3")
        h_conv6_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv6_bn_b3_2, W_con6_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_con6_b3_3)
        h_conv6_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv6_b3_3, layer_depth=96, is_training=is_training)

        h_conv6_bn_mp = tf.nn.max_pool(h_conv5_bn_mp, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_con6_b4_1 = tf.Variable(tf.truncated_normal([1, 1, 192, 32], stddev=0.1), trainable=True, name="W_Mixed_5b_b4_1")
        b_con6_b4_1 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True, name="b_Mixed_5b_b4_1")
        h_conv6_b4_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv6_bn_mp, W_con6_b4_1, strides=[1, 1, 1, 1], padding="SAME"), b_con6_b4_1)
        h_conv6_bn_b4_1 = conv_batch_normalization(prev_layer=h_conv6_b4_1, layer_depth=32, is_training=is_training)
        
        h_conv6 = tf.concat([h_conv6_bn_b1_1, h_conv6_bn_b2_2, h_conv6_bn_b3_3, h_conv6_bn_b4_1], 3)
        
    with tf.name_scope("Mixed_5c"):
        W_conv7_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1), trainable=True, name="W_Mixed_5c_b1_1")
        b_conv7_b1_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5c_b1_1")
        h_conv7_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv6, W_conv7_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv7_b1_1)
        h_conv7_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv7_b1_1, layer_depth=64, is_training=is_training)

        W_conv7_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 256, 48], stddev=0.1), trainable=True, name="W_Mixed_5c_b2_1")
        b_conv7_b2_1 = tf.Variable(tf.constant(0.0, shape=[48]), trainable=True, name="b_Mixed_5c_b2_1")
        h_conv7_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv6, W_conv7_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv7_b2_1)
        h_conv7_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv7_b2_1, layer_depth=48, is_training=is_training)
        W_conv7_b2_2 = tf.Variable(tf.truncated_normal([5, 5, 48, 64], stddev=0.1), trainable=True, name="W_Mixed_5c_b2_2")
        b_conv7_b2_2 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5c_b2_2")
        h_conv7_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv7_bn_b2_1, W_conv7_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv7_b2_2)
        h_conv7_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv7_b2_2, layer_depth=64, is_training=is_training)

        W_conv7_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1), trainable=True,name="W_Mixed_5c_b3_1")
        b_conv7_b3_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5c_b3_1")
        h_conv7_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv6, W_conv7_b3_1, strides=[1, 1, 1, 1], padding="SAME"),b_conv7_b3_1)
        h_conv7_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv7_b3_1, layer_depth=64, is_training=is_training)
        W_conv7_b3_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1), trainable=True,name="W_Mixed_5c_b3_2")
        b_conv7_b3_2 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5c_b3_2")
        h_conv7_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv7_bn_b3_1, W_conv7_b3_2, strides=[1, 1, 1, 1], padding="SAME"),b_conv7_b3_2)
        h_conv7_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv7_b3_2, layer_depth=96, is_training=is_training)
        W_conv7_b3_3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1), trainable=True,name="W_Mixed_5c_b3_3")
        b_conv7_b3_3 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5c_b3_3")
        h_conv7_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv7_bn_b3_2, W_conv7_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv7_b3_3)
        h_conv7_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv7_b3_3, layer_depth=96, is_training=is_training)

        h_conv7_bn_mp = tf.nn.max_pool(h_conv6, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv7_b4 = tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1), trainable=True, name="W_Mixed_5c_b4_4")
        b_conv7_b4 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5c_b4_4")
        h_conv7_b4 = tf.nn.bias_add(tf.nn.conv2d(h_conv7_bn_mp, W_conv7_b4, strides=[1, 1, 1, 1], padding="SAME"), b_conv7_b4)
        h_conv7_bn_b4 = conv_batch_normalization(prev_layer=h_conv7_b4, layer_depth=64, is_training=is_training)

        h_conv7 = tf.concat([h_conv7_bn_b1_1, h_conv7_bn_b2_2, h_conv7_bn_b3_3, h_conv7_bn_b4], 3)

    with tf.name_scope("Mixed_5d"):
        W_conv8_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 288, 64], stddev=0.1), trainable=True, name="W_Mixed_5d_b1_1")
        b_conv8_b1_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5d_b1_1")
        h_conv8_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv7, W_conv8_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv8_b1_1)
        h_conv8_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv8_b1_1, layer_depth=64, is_training=is_training)

        W_conv8_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 288, 48], stddev=0.1), trainable=True, name="W_Mixed_5d_b2_1")
        b_conv8_b2_1 = tf.Variable(tf.constant(0.0, shape=[48]), trainable=True, name="b_Mixed_5d_b2_1")
        h_conv8_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv7, W_conv8_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv8_b2_1)
        h_conv8_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv8_b2_1, layer_depth=48, is_training=is_training)
        W_conv8_b2_2 = tf.Variable(tf.truncated_normal([5, 5, 48, 64], stddev=0.1), trainable=True, name="W_Mixed_5d_b2_2")
        b_conv8_b2_2 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5d_b2_2")
        h_conv8_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv8_bn_b2_1, W_conv8_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv8_b2_2)
        h_conv8_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv8_b2_2, layer_depth=64, is_training=is_training)

        W_conv8_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 288, 64], stddev=0.1), trainable=True,name="W_Mixed_5d_b3_1")
        b_conv8_b3_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5d_b3_1")
        h_conv8_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv7, W_conv8_b3_1, strides=[1, 1, 1, 1], padding="SAME"),b_conv8_b3_1)
        h_conv8_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv8_b3_1, layer_depth=64, is_training=is_training)
        W_conv8_b3_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1), trainable=True,name="W_Mixed_5d_b3_2")
        b_conv8_b3_2 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5d_b3_2")
        h_conv8_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv8_bn_b3_1, W_conv8_b3_2, strides=[1, 1, 1, 1], padding="SAME"),b_conv8_b3_2)
        h_conv8_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv8_b3_2, layer_depth=96, is_training=is_training)
        W_conv8_b3_3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1), trainable=True,name="W_Mixed_5d_b3_3")
        b_conv8_b3_3 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_5d_b3_3")
        h_conv8_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv8_bn_b3_2, W_conv8_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv8_b3_3)
        h_conv8_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv8_b3_3, layer_depth=96, is_training=is_training)

        h_conv8_bn_mp = tf.nn.max_pool(h_conv7, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv8_b4 = tf.Variable(tf.truncated_normal([1, 1, 288, 64], stddev=0.1), trainable=True, name="W_Mixed_5d_b4_4")
        b_conv8_b4 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_5d_b4_4")
        h_conv8_b4 = tf.nn.bias_add(tf.nn.conv2d(h_conv8_bn_mp, W_conv8_b4, strides=[1, 1, 1, 1], padding="SAME"), b_conv8_b4)
        h_conv8_bn_b4 = conv_batch_normalization(prev_layer=h_conv8_b4, layer_depth=64, is_training=is_training)

        h_conv8 = tf.concat([h_conv8_bn_b1_1, h_conv8_bn_b2_2, h_conv8_bn_b3_3, h_conv8_bn_b4], 3)
        
    with tf.name_scope("Mixed_6a"):
        W_conv9_b1_1 = tf.Variable(tf.truncated_normal([3, 3, 288, 384], stddev=0.1), trainable=True, name="W_Mixed_6a_b1_1")
        b_conv9_b1_1 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_6a_b1_1")
        h_conv9_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv8, W_conv9_b1_1, strides=[1, 2, 2, 1], padding="VALID"), b_conv9_b1_1)
        h_conv9_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv9_b1_1, layer_depth=384, is_training=is_training)

        W_conv9_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 288, 64], stddev=0.1), trainable=True, name="W_Mixed_6a_b2_1")
        b_conv9_b2_1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_Mixed_6a_b2_1")
        h_conv9_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv8, W_conv9_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv9_b2_1)
        h_conv9_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv9_b2_1, layer_depth=64, is_training=is_training)
        W_conv9_b2_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1), trainable=True, name="W_Mixed_6a_b2_2")
        b_conv9_b2_2 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_6a_b2_2")
        h_conv9_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv9_bn_b2_1, W_conv9_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv9_b2_2)
        h_conv9_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv9_b2_2, layer_depth=96, is_training=is_training)
        W_conv9_b2_3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1), trainable=True,name="W_Mixed_6a_b2_3")
        b_conv9_b2_3 = tf.Variable(tf.constant(0.0, shape=[96]), trainable=True, name="b_Mixed_6a_b2_3")
        h_conv9_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv9_bn_b2_2, W_conv9_b2_3, strides=[1, 2, 2, 1], padding="VALID"), b_conv9_b2_3)
        h_conv9_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv9_b2_3, layer_depth=96, is_training=is_training)

        h_conv9_bn_mp = tf.nn.max_pool(h_conv8, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        h_conv9 = tf.concat([h_conv9_bn_b1_1, h_conv9_bn_b2_3, h_conv9_bn_mp], 3)
        
    with tf.name_scope("Mixed_6b"):
        W_conv10_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6b_b1_1")
        b_conv10_b1_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6b_b1_1")
        h_conv10_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv9, W_conv10_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b1_1)
        h_conv10_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv10_b1_1, layer_depth=192, is_training=is_training)

        W_conv10_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 128], stddev=0.1), trainable=True, name="W_Mixed_6b_b2_1")
        b_conv10_b2_1 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b2_1")
        h_conv10_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv9, W_conv10_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b2_1)
        h_conv10_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv10_b2_1, layer_depth=128, is_training=is_training)
        W_conv10_b2_2 = tf.Variable(tf.truncated_normal([1, 7, 128, 128], stddev=0.1), trainable=True, name="W_Mixed_6b_b2_2")
        b_conv10_b2_2 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b2_2")
        h_conv10_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b2_1, W_conv10_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b2_2)
        h_conv10_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv10_b2_2, layer_depth=128, is_training=is_training)
        W_conv10_b2_3 = tf.Variable(tf.truncated_normal([7, 1, 128, 192], stddev=0.1), trainable=True,name="W_Mixed_6b_b2_3")
        b_conv10_b2_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6b_b2_3")
        h_conv10_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b2_2, W_conv10_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b2_3)
        h_conv10_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv10_b2_3, layer_depth=192, is_training=is_training)

        W_conv10_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 128], stddev=0.1), trainable=True, name="W_Mixed_6b_b3_1")
        b_conv10_b3_1 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b3_1")
        h_conv10_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv9, W_conv10_b3_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b3_1)
        h_conv10_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv10_b3_1, layer_depth=128, is_training=is_training)
        W_conv10_b3_2 = tf.Variable(tf.truncated_normal([7, 1, 128, 128], stddev=0.1), trainable=True, name="W_Mixed_6b_b3_2")
        b_conv10_b3_2 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b3_2")
        h_conv10_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b3_1, W_conv10_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b3_2)
        h_conv10_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv10_b3_2, layer_depth=128, is_training=is_training)
        W_conv10_b3_3 = tf.Variable(tf.truncated_normal([1, 7, 128, 128], stddev=0.1), trainable=True, name="W_Mixed_6b_b3_3")
        b_conv10_b3_3 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b3_3")
        h_conv10_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b3_2, W_conv10_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b3_3)
        h_conv10_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv10_b3_3, layer_depth=128, is_training=is_training)
        W_conv10_b3_4 = tf.Variable(tf.truncated_normal([7, 1, 128, 128], stddev=0.1), trainable=True,name="W_Mixed_6b_b3_4")
        b_conv10_b3_4 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_Mixed_6b_b3_4")
        h_conv10_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b3_3, W_conv10_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b3_4)
        h_conv10_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv10_b3_4, layer_depth=128, is_training=is_training)
        W_conv10_b3_5 = tf.Variable(tf.truncated_normal([1, 7, 128, 192], stddev=0.1), trainable=True,name="W_Mixed_6b_b3_5")
        b_conv10_b3_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6b_b3_5")
        h_conv10_b3_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_b3_4, W_conv10_b3_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b3_5)
        h_conv10_bn_b3_5 = conv_batch_normalization(prev_layer=h_conv10_b3_5, layer_depth=192, is_training=is_training)

        h_conv10_bn_mp = tf.nn.avg_pool(h_conv9, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv10_b4_5 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True,name="W_Mixed_6b_b4_5")
        b_conv10_b4_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6b_b4_5")
        h_conv10_b4_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv10_bn_mp, W_conv10_b4_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv10_b4_5)
        h_conv10_bn_b4_5 = conv_batch_normalization(prev_layer=h_conv10_b4_5, layer_depth=192, is_training=is_training)

        h_conv10 = tf.concat([h_conv10_bn_b1_1, h_conv10_bn_b2_3, h_conv10_bn_b3_5, h_conv10_bn_b4_5], 3)
        
    with tf.name_scope("Mixed_6c"):
        W_conv11_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6c_b1_1")
        b_conv11_b1_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6c_b1_1")
        h_conv11_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv10, W_conv11_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b1_1)
        h_conv11_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv11_b1_1, layer_depth=192, is_training=is_training)

        W_conv11_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 160], stddev=0.1), trainable=True, name="W_Mixed_6c_b2_1")
        b_conv11_b2_1 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b2_1")
        h_conv11_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv10, W_conv11_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b2_1)
        h_conv11_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv11_b2_1, layer_depth=160, is_training=is_training)
        W_conv11_b2_2 = tf.Variable(tf.truncated_normal([1, 7, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6c_b2_2")
        b_conv11_b2_2 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b2_2")
        h_conv11_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b2_1, W_conv11_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b2_2)
        h_conv11_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv11_b2_2, layer_depth=160, is_training=is_training)
        W_conv11_b2_3 = tf.Variable(tf.truncated_normal([7, 1, 160, 192], stddev=0.1), trainable=True,name="W_Mixed_6c_b2_3")
        b_conv11_b2_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6c_b2_3")
        h_conv11_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b2_2, W_conv11_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b2_3)
        h_conv11_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv11_b2_3, layer_depth=192, is_training=is_training)

        W_conv11_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 160], stddev=0.1), trainable=True, name="W_Mixed_6c_b3_1")
        b_conv11_b3_1 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b3_1")
        h_conv11_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv10, W_conv11_b3_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b3_1)
        h_conv11_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv11_b3_1, layer_depth=160, is_training=is_training)
        W_conv11_b3_2 = tf.Variable(tf.truncated_normal([7, 1, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6c_b3_2")
        b_conv11_b3_2 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b3_2")
        h_conv11_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b3_1, W_conv11_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b3_2)
        h_conv11_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv11_b3_2, layer_depth=160, is_training=is_training)
        W_conv11_b3_3 = tf.Variable(tf.truncated_normal([1, 7, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6c_b3_3")
        b_conv11_b3_3 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b3_3")
        h_conv11_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b3_2, W_conv11_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b3_3)
        h_conv11_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv11_b3_3, layer_depth=160, is_training=is_training)
        W_conv11_b3_4 = tf.Variable(tf.truncated_normal([7, 1, 160, 160], stddev=0.1), trainable=True,name="W_Mixed_6c_b3_4")
        b_conv11_b3_4 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6c_b3_4")
        h_conv11_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b3_3, W_conv11_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b3_4)
        h_conv11_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv11_b3_4, layer_depth=160, is_training=is_training)
        W_conv11_b3_5 = tf.Variable(tf.truncated_normal([1, 7, 160, 192], stddev=0.1), trainable=True,name="W_Mixed_6c_b3_5")
        b_conv11_b3_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6c_b3_5")
        h_conv11_b3_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_b3_4, W_conv11_b3_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b3_5)
        h_conv11_bn_b3_5 = conv_batch_normalization(prev_layer=h_conv11_b3_5, layer_depth=192, is_training=is_training)

        h_conv11_bn_mp = tf.nn.avg_pool(h_conv10, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv11_b4_5 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True,name="W_Mixed_6c_b4_5")
        b_conv11_b4_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6c_b4_5")
        h_conv11_b4_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv11_bn_mp, W_conv11_b4_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv11_b4_5)
        h_conv11_bn_b4_5 = conv_batch_normalization(prev_layer=h_conv11_b4_5, layer_depth=192, is_training=is_training)

        h_conv11 = tf.concat([h_conv11_bn_b1_1, h_conv11_bn_b2_3, h_conv11_bn_b3_5, h_conv11_bn_b4_5], 3)
        
    with tf.name_scope("Mixed_6d"):
        W_conv12_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6d_b1_1")
        b_conv12_b1_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6d_b1_1")
        h_conv12_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv11, W_conv12_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b1_1)
        h_conv12_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv12_b1_1, layer_depth=192, is_training=is_training)

        W_conv12_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 160], stddev=0.1), trainable=True, name="W_Mixed_6d_b2_1")
        b_conv12_b2_1 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b2_1")
        h_conv12_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv11, W_conv12_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b2_1)
        h_conv12_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv12_b2_1, layer_depth=160, is_training=is_training)
        W_conv12_b2_2 = tf.Variable(tf.truncated_normal([1, 7, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6d_b2_2")
        b_conv12_b2_2 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b2_2")
        h_conv12_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b2_1, W_conv12_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b2_2)
        h_conv12_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv12_b2_2, layer_depth=160, is_training=is_training)
        W_conv12_b2_3 = tf.Variable(tf.truncated_normal([7, 1, 160, 192], stddev=0.1), trainable=True,name="W_Mixed_6d_b2_3")
        b_conv12_b2_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6d_b2_3")
        h_conv12_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b2_2, W_conv12_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b2_3)
        h_conv12_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv12_b2_3, layer_depth=192, is_training=is_training)

        W_conv12_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 160], stddev=0.1), trainable=True, name="W_Mixed_6d_b3_1")
        b_conv12_b3_1 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b3_1")
        h_conv12_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv11, W_conv12_b3_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b3_1)
        h_conv12_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv12_b3_1, layer_depth=160, is_training=is_training)
        W_conv12_b3_2 = tf.Variable(tf.truncated_normal([7, 1, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6d_b3_2")
        b_conv12_b3_2 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b3_2")
        h_conv12_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b3_1, W_conv12_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b3_2)
        h_conv12_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv12_b3_2, layer_depth=160, is_training=is_training)
        W_conv12_b3_3 = tf.Variable(tf.truncated_normal([1, 7, 160, 160], stddev=0.1), trainable=True, name="W_Mixed_6d_b3_3")
        b_conv12_b3_3 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b3_3")
        h_conv12_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b3_2, W_conv12_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b3_3)
        h_conv12_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv12_b3_3, layer_depth=160, is_training=is_training)
        W_conv12_b3_4 = tf.Variable(tf.truncated_normal([7, 1, 160, 160], stddev=0.1), trainable=True,name="W_Mixed_6d_b3_4")
        b_conv12_b3_4 = tf.Variable(tf.constant(0.0, shape=[160]), trainable=True, name="b_Mixed_6d_b3_4")
        h_conv12_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b3_3, W_conv12_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b3_4)
        h_conv12_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv12_b3_4, layer_depth=160, is_training=is_training)
        W_conv12_b3_5 = tf.Variable(tf.truncated_normal([1, 7, 160, 192], stddev=0.1), trainable=True,name="W_Mixed_6d_b3_5")
        b_conv12_b3_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6d_b3_5")
        h_conv12_b3_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_b3_4, W_conv12_b3_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b3_5)
        h_conv12_bn_b3_5 = conv_batch_normalization(prev_layer=h_conv12_b3_5, layer_depth=192, is_training=is_training)

        h_conv12_bn_mp = tf.nn.avg_pool(h_conv11, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv12_b4_5 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True,name="W_Mixed_6d_b4_5")
        b_conv12_b4_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6d_b4_5")
        h_conv12_b4_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv12_bn_mp, W_conv12_b4_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv12_b4_5)
        h_conv12_bn_b4_5 = conv_batch_normalization(prev_layer=h_conv12_b4_5, layer_depth=192, is_training=is_training)

        h_conv12 = tf.concat([h_conv12_bn_b1_1, h_conv12_bn_b2_3, h_conv12_bn_b3_5, h_conv12_bn_b4_5], 3)
        
    with tf.name_scope("Mixed_6e"):
        W_conv13_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b1_1")
        b_conv13_b1_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b1_1")
        h_conv13_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv12, W_conv13_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b1_1)
        h_conv13_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv13_b1_1, layer_depth=192, is_training=is_training)

        W_conv13_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b2_1")
        b_conv13_b2_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b2_1")
        h_conv13_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv12, W_conv13_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b2_1)
        h_conv13_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv13_b2_1, layer_depth=192, is_training=is_training)
        W_conv13_b2_2 = tf.Variable(tf.truncated_normal([1, 7, 192, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b2_2")
        b_conv13_b2_2 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b2_2")
        h_conv13_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b2_1, W_conv13_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b2_2)
        h_conv13_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv13_b2_2, layer_depth=192, is_training=is_training)
        W_conv13_b2_3 = tf.Variable(tf.truncated_normal([7, 1, 192, 192], stddev=0.1), trainable=True,name="W_Mixed_6e_b2_3")
        b_conv13_b2_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b2_3")
        h_conv13_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b2_2, W_conv13_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b2_3)
        h_conv13_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv13_b2_3, layer_depth=192, is_training=is_training)

        W_conv13_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b3_1")
        b_conv13_b3_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b3_1")
        h_conv13_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv12, W_conv13_b3_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b3_1)
        h_conv13_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv13_b3_1, layer_depth=192, is_training=is_training)
        W_conv13_b3_2 = tf.Variable(tf.truncated_normal([7, 1, 192, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b3_2")
        b_conv13_b3_2 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b3_2")
        h_conv13_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b3_1, W_conv13_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b3_2)
        h_conv13_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv13_b3_2, layer_depth=192, is_training=is_training)
        W_conv13_b3_3 = tf.Variable(tf.truncated_normal([1, 7, 192, 192], stddev=0.1), trainable=True, name="W_Mixed_6e_b3_3")
        b_conv13_b3_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b3_3")
        h_conv13_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b3_2, W_conv13_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b3_3)
        h_conv13_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv13_b3_3, layer_depth=192, is_training=is_training)
        W_conv13_b3_4 = tf.Variable(tf.truncated_normal([7, 1, 192, 192], stddev=0.1), trainable=True,name="W_Mixed_6e_b3_4")
        b_conv13_b3_4 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b3_4")
        h_conv13_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b3_3, W_conv13_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b3_4)
        h_conv13_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv13_b3_4, layer_depth=192, is_training=is_training)
        W_conv13_b3_5 = tf.Variable(tf.truncated_normal([1, 7, 192, 192], stddev=0.1), trainable=True,name="W_Mixed_6e_b3_5")
        b_conv13_b3_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b3_5")
        h_conv13_b3_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_b3_4, W_conv13_b3_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b3_5)
        h_conv13_bn_b3_5 = conv_batch_normalization(prev_layer=h_conv13_b3_5, layer_depth=192, is_training=is_training)

        h_conv13_bn_mp = tf.nn.avg_pool(h_conv12, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv13_b4_5 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True,name="W_Mixed_6e_b4_5")
        b_conv13_b4_5 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_6e_b4_5")
        h_conv13_b4_5 = tf.nn.bias_add(tf.nn.conv2d(h_conv13_bn_mp, W_conv13_b4_5, strides=[1, 1, 1, 1], padding="SAME"), b_conv13_b4_5)
        h_conv13_bn_b4_5 = conv_batch_normalization(prev_layer=h_conv13_b4_5, layer_depth=192, is_training=is_training)

        h_conv13 = tf.concat([h_conv13_bn_b1_1, h_conv13_bn_b2_3, h_conv13_bn_b3_5, h_conv13_bn_b4_5], 3)
        
    with tf.name_scope("Mixed_7a"):
        W_conv14_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_7a_b1_1")
        b_conv14_b1_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7a_b1_1")
        h_conv14_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv13, W_conv14_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv14_b1_1)
        h_conv14_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv14_b1_1, layer_depth=192, is_training=is_training)
        W_conv14_b1_2 = tf.Variable(tf.truncated_normal([3, 3, 192, 320], stddev=0.1), trainable=True,name="W_Mixed_7a_b1_2")
        b_conv14_b1_2 = tf.Variable(tf.constant(0.0, shape=[320]), trainable=True, name="b_Mixed_7a_b1_2")
        h_conv14_b1_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv14_bn_b1_1, W_conv14_b1_2, strides=[1, 2, 2, 1], padding="VALID"),b_conv14_b1_2)
        h_conv14_bn_b1_2 = conv_batch_normalization(prev_layer=h_conv14_b1_2, layer_depth=320, is_training=is_training)

        W_conv14_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 768, 192], stddev=0.1), trainable=True, name="W_Mixed_7a_b2_1")
        b_conv14_b2_1 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7a_b2_1")
        h_conv14_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv13, W_conv14_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv14_b2_1)
        h_conv14_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv14_b2_1, layer_depth=192, is_training=is_training)
        W_conv14_b2_2 = tf.Variable(tf.truncated_normal([1, 7, 192, 192], stddev=0.1), trainable=True, name="W_Mixed_7a_b2_2")
        b_conv14_b2_2 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7a_b2_2")
        h_conv14_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv14_bn_b2_1, W_conv14_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv14_b2_2)
        h_conv14_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv14_b2_2, layer_depth=192, is_training=is_training)
        W_conv14_b2_3 = tf.Variable(tf.truncated_normal([7, 1, 192, 192], stddev=0.1), trainable=True,name="W_Mixed_7a_b2_3")
        b_conv14_b2_3 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7a_b2_3")
        h_conv14_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv14_bn_b2_2, W_conv14_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv14_b2_3)
        h_conv14_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv14_b2_3, layer_depth=192, is_training=is_training)
        W_conv14_b2_4 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=0.1), trainable=True,name="W_Mixed_7a_b2_4")
        b_conv14_b2_4 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7a_b2_4")
        h_conv14_b2_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv14_bn_b2_3, W_conv14_b2_4, strides=[1, 2, 2, 1], padding="VALID"), b_conv14_b2_4)
        h_conv14_bn_b2_4 = conv_batch_normalization(prev_layer=h_conv14_b2_4, layer_depth=192, is_training=is_training)

        h_conv14_bn_mp = tf.nn.avg_pool(h_conv13, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        h_conv14 = tf.concat([h_conv14_bn_b1_2, h_conv14_bn_b2_4, h_conv14_bn_mp], 3)
        
    with tf.name_scope("Mixed_7b"):
        W_conv15_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 1280, 320], stddev=0.1), trainable=True, name="W_Mixed_7b_b1_1")
        b_conv15_b1_1 = tf.Variable(tf.constant(0.0, shape=[320]), trainable=True, name="b_Mixed_7b_b1_1")
        h_conv15_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv14, W_conv15_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b1_1)
        h_conv15_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv15_b1_1, layer_depth=320, is_training=is_training)

        W_conv15_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 1280, 384], stddev=0.1), trainable=True, name="W_Mixed_7b_b2_1")
        b_conv15_b2_1 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b2_1")
        h_conv15_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv14, W_conv15_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b2_1)
        h_conv15_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv15_b2_1, layer_depth=384, is_training=is_training)
        W_conv15_b2_2 = tf.Variable(tf.truncated_normal([1, 3, 384, 384], stddev=0.1), trainable=True, name="W_Mixed_7b_b2_2")
        b_conv15_b2_2 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b2_2")
        h_conv15_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_b2_1, W_conv15_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b2_2)
        h_conv15_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv15_b2_2, layer_depth=384, is_training=is_training)
        W_conv15_b2_3 = tf.Variable(tf.truncated_normal([3, 1, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7b_b2_3")
        b_conv15_b2_3 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b2_3")
        h_conv15_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_b2_1, W_conv15_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b2_3)
        h_conv15_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv15_b2_3, layer_depth=384, is_training=is_training)
        h_conv15_bn_b2_4 = tf.concat([h_conv15_bn_b2_2, h_conv15_bn_b2_3], 3)

        W_conv15_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 1280, 448], stddev=0.1), trainable=True,name="W_Mixed_7b_b3_1")
        b_conv15_b3_1 = tf.Variable(tf.constant(0.0, shape=[448]), trainable=True, name="b_Mixed_7b_b3_1")
        h_conv15_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv14, W_conv15_b3_1, strides=[1, 1, 1, 1], padding="SAME"),b_conv15_b3_1)
        h_conv15_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv15_b3_1, layer_depth=448, is_training=is_training)
        W_conv15_b3_2 = tf.Variable(tf.truncated_normal([3, 3, 448, 384], stddev=0.1), trainable=True,name="W_Mixed_7b_b3_2")
        b_conv15_b3_2 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b3_2")
        h_conv15_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_b3_1, W_conv15_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b3_2)
        h_conv15_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv15_b3_2, layer_depth=384, is_training=is_training)
        W_conv15_b3_3 = tf.Variable(tf.truncated_normal([1, 3, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7b_b3_3")
        b_conv15_b3_3 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b3_3")
        h_conv15_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_b3_2, W_conv15_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b3_3)
        h_conv15_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv15_b3_3, layer_depth=384, is_training=is_training)
        W_conv15_b3_4 = tf.Variable(tf.truncated_normal([3, 1, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7b_b3_4")
        b_conv15_b3_4 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7b_b3_4")
        h_conv15_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_b3_2, W_conv15_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b3_4)
        h_conv15_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv15_b3_4, layer_depth=384, is_training=is_training)
        h_conv15_bn_b3_5 = tf.concat([h_conv15_bn_b3_3, h_conv15_bn_b3_4], 3)

        h_conv15_bn_mp = tf.nn.avg_pool(h_conv14, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv15_b4_4 = tf.Variable(tf.truncated_normal([1, 1, 1280, 192], stddev=0.1), trainable=True,name="W_Mixed_7b_b4_4")
        b_conv15_b4_4 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7b_b4_4")
        h_conv15_b4_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv15_bn_mp, W_conv15_b4_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv15_b4_4)
        h_conv15_bn_b4_4 = conv_batch_normalization(prev_layer=h_conv15_b4_4, layer_depth=192, is_training=is_training)

        h_conv15 = tf.concat([h_conv15_bn_b1_1, h_conv15_bn_b2_4, h_conv15_bn_b3_5, h_conv15_bn_b4_4], 3)
        
    with tf.name_scope("Mixed_7c"):
        W_conv16_b1_1 = tf.Variable(tf.truncated_normal([1, 1, 2048, 320], stddev=0.1), trainable=True, name="W_Mixed_7c_b1_1")
        b_conv16_b1_1 = tf.Variable(tf.constant(0.0, shape=[320]), trainable=True, name="b_Mixed_7c_b1_1")
        h_conv16_b1_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv15, W_conv16_b1_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b1_1)
        h_conv16_bn_b1_1 = conv_batch_normalization(prev_layer=h_conv16_b1_1, layer_depth=320, is_training=is_training)

        W_conv16_b2_1 = tf.Variable(tf.truncated_normal([1, 1, 2048, 384], stddev=0.1), trainable=True, name="W_Mixed_7c_b2_1")
        b_conv16_b2_1 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b2_1")
        h_conv16_b2_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv15, W_conv16_b2_1, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b2_1)
        h_conv16_bn_b2_1 = conv_batch_normalization(prev_layer=h_conv16_b2_1, layer_depth=384, is_training=is_training)
        W_conv16_b2_2 = tf.Variable(tf.truncated_normal([1, 3, 384, 384], stddev=0.1), trainable=True, name="W_Mixed_7c_b2_2")
        b_conv16_b2_2 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b2_2")
        h_conv16_b2_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_b2_1, W_conv16_b2_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b2_2)
        h_conv16_bn_b2_2 = conv_batch_normalization(prev_layer=h_conv16_b2_2, layer_depth=384, is_training=is_training)
        W_conv16_b2_3 = tf.Variable(tf.truncated_normal([3, 1, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7c_b2_3")
        b_conv16_b2_3 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b2_3")
        h_conv16_b2_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_b2_1, W_conv16_b2_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b2_3)
        h_conv16_bn_b2_3 = conv_batch_normalization(prev_layer=h_conv16_b2_3, layer_depth=384, is_training=is_training)
        h_conv16_bn_b2_4 = tf.concat([h_conv16_bn_b2_2, h_conv16_bn_b2_3], 3)

        W_conv16_b3_1 = tf.Variable(tf.truncated_normal([1, 1, 2048, 448], stddev=0.1), trainable=True,name="W_Mixed_7c_b3_1")
        b_conv16_b3_1 = tf.Variable(tf.constant(0.0, shape=[448]), trainable=True, name="b_Mixed_7c_b3_1")
        h_conv16_b3_1 = tf.nn.bias_add(tf.nn.conv2d(h_conv15, W_conv16_b3_1, strides=[1, 1, 1, 1], padding="SAME"),b_conv16_b3_1)
        h_conv16_bn_b3_1 = conv_batch_normalization(prev_layer=h_conv16_b3_1, layer_depth=448, is_training=is_training)
        W_conv16_b3_2 = tf.Variable(tf.truncated_normal([3, 3, 448, 384], stddev=0.1), trainable=True,name="W_Mixed_7c_b3_2")
        b_conv16_b3_2 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b3_2")
        h_conv16_b3_2 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_b3_1, W_conv16_b3_2, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b3_2)
        h_conv16_bn_b3_2 = conv_batch_normalization(prev_layer=h_conv16_b3_2, layer_depth=384, is_training=is_training)
        W_conv16_b3_3 = tf.Variable(tf.truncated_normal([1, 3, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7c_b3_3")
        b_conv16_b3_3 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b3_3")
        h_conv16_b3_3 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_b3_2, W_conv16_b3_3, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b3_3)
        h_conv16_bn_b3_3 = conv_batch_normalization(prev_layer=h_conv16_b3_3, layer_depth=384, is_training=is_training)
        W_conv16_b3_4 = tf.Variable(tf.truncated_normal([3, 1, 384, 384], stddev=0.1), trainable=True,name="W_Mixed_7c_b3_4")
        b_conv16_b3_4 = tf.Variable(tf.constant(0.0, shape=[384]), trainable=True, name="b_Mixed_7c_b3_4")
        h_conv16_b3_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_b3_2, W_conv16_b3_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b3_4)
        h_conv16_bn_b3_4 = conv_batch_normalization(prev_layer=h_conv16_b3_4, layer_depth=384, is_training=is_training)
        h_conv16_bn_b3_5 = tf.concat([h_conv16_bn_b3_3, h_conv16_bn_b3_4], 3)

        h_conv16_bn_mp = tf.nn.avg_pool(h_conv15, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        W_conv16_b4_4 = tf.Variable(tf.truncated_normal([1, 1, 2048, 192], stddev=0.1), trainable=True,name="W_Mixed_7c_b4_4")
        b_conv16_b4_4 = tf.Variable(tf.constant(0.0, shape=[192]), trainable=True, name="b_Mixed_7c_b4_4")
        h_conv16_b4_4 = tf.nn.bias_add(tf.nn.conv2d(h_conv16_bn_mp, W_conv16_b4_4, strides=[1, 1, 1, 1], padding="SAME"), b_conv16_b4_4)
        h_conv16_bn_b4_4 = conv_batch_normalization(prev_layer=h_conv16_b4_4, layer_depth=192, is_training=is_training)

        h_conv16 = tf.concat([h_conv16_bn_b1_1, h_conv16_bn_b2_4, h_conv16_bn_b3_5, h_conv16_bn_b4_4], 3)

    with tf.name_scope("output"):
        inception_output = tf.nn.avg_pool(h_conv16, [1, 8, 8, 1], [1, 1, 1, 1], padding="VALID")
        rate = tf.placeholder(tf.float32, name="rate")
        inception_output_dropout = tf.nn.dropout(inception_output, rate=rate)
        W_conv17 = tf.Variable(tf.truncated_normal([1, 1, 2048, 5], stddev=0.1), trainable=True,name="W_output")
        b_conv17 = tf.Variable(tf.constant(0.0, shape=[5]), trainable=True, name="b_output")
        output = tf.nn.bias_add(tf.nn.conv2d(inception_output_dropout, W_conv17, strides=[1, 1, 1, 1], padding="SAME"), b_conv17)
        output_squeezed = tf.squeeze(output, [1, 2])


with tf.name_scope("loss"):
    one_hot_labels = slim.one_hot_encoding(y_, classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_squeezed, labels=one_hot_labels))
    total_loss = loss

with tf.name_scope("train"):
    lr = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=total_loss)

update_learning_rate = tf.assign(lr, lr * 0.8)

correct_prediction = tf.equal(y_, tf.argmax(output_squeezed, 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # ckpt = tf.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError('The ckpt file is None.')
    dataset_train = tf.data.TFRecordDataset([train_tfrecord])
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)

    for epoch in range(epochs):
        if epoch > 100 and epoch % 10 == 0:
            sess.run(update_learning_rate)
        print("learning_rate:", sess.run(lr))
        for step in range(int(train_size/batch_size)):
            img_train, label_train, filename = sess.run(next_element_train)
            # cv2.imshow("image", np.squeeze(img_train))
            # filename = filename[0].decode()
            # print(label_train, filename)
            # cv2.waitKey(0)
            _, _loss, _accuracy = sess.run([train_step, loss, accuracy], feed_dict={x_: img_train, y_: label_train, is_training:True, rate:0.2})
            if step % 10 == 0:
                print("step:", step / 10, " loss:", _loss, " accuracy:", _accuracy)
        tf.train.Saver().save(sess, "ckpt/model.ckpt")
        print("save ckpt:", epoch)
    # saver.export_meta_graph("ckpt/model.meta")
    # tf.summary.FileWriter("logs/", sess.graph)
    # tf.train.write_graph(sess.graph_def, "", "graph.pbtxt")
    # with tf.gfile.GFile("graph.pb", mode="wb") as f:
    #     f.write(convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"]).SerializeToString())
