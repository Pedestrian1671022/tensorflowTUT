import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

image_pixels = 224
batch_size = 20
classes = 5
epochs = 200
train_size = 10935
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
    return batch_normalized_output

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

with tf.name_scope("Resnet50"):
    with tf.name_scope("conv1"):
        # x_1 = tf.pad(x_, [[0, 0], [3, 3], [3, 3], [0, 0]])
        W_con1 = tf.Variable(tf.truncated_normal([7, 7, 3, 64], stddev=0.1), trainable=True, name="W_conv1")
        b_con1 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv1")
        h_conv1 = tf.nn.bias_add(tf.nn.conv2d(x_, W_con1, strides=[1, 2, 2, 1], padding="VALID"), b_con1)
        h_conv1_bn = conv_batch_normalization(prev_layer=h_conv1, layer_depth=64, is_training=is_training)
        h_conv1_bn_relu = tf.nn.relu(h_conv1_bn)
        h_conv1_mp = tf.nn.max_pool(h_conv1_bn_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
        
    with tf.name_scope("conv2_1_a"):
        W_conv2_1_a = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=0.1), trainable=True, name="W_conv2_1_a")
        b_conv2_1_a = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_1_a")
        h_conv2_1_a = tf.nn.bias_add(tf.nn.conv2d(h_conv1_mp, W_conv2_1_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_1_a)
        h_conv2_1_a_bn = conv_batch_normalization(prev_layer=h_conv2_1_a, layer_depth=64, is_training=is_training)
        h_conv2_1_a_bn_relu = tf.nn.relu(h_conv2_1_a_bn)
    with tf.name_scope("conv2_1_b"):
        W_conv2_1_b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), trainable=True, name="W_conv2_1_b")
        b_conv2_1_b = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_1_b")
        h_conv2_1_b = tf.nn.bias_add(tf.nn.conv2d(h_conv2_1_a_bn_relu, W_conv2_1_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_1_b)
        h_conv2_1_b_bn = conv_batch_normalization(prev_layer=h_conv2_1_b, layer_depth=64, is_training=is_training)
        h_conv2_1_b_bn_relu = tf.nn.relu(h_conv2_1_b_bn)
    with tf.name_scope("conv2_1_c"):
        W_conv2_1_c = tf.Variable(tf.truncated_normal([1, 1, 64, 256], stddev=0.1), trainable=True, name="W_conv2_1_c")
        b_conv2_1_c = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv2_1_c")
        h_conv2_1_c = tf.nn.bias_add(tf.nn.conv2d(h_conv2_1_b_bn_relu, W_conv2_1_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_1_c)
        h_conv2_1_c_bn = conv_batch_normalization(prev_layer=h_conv2_1_c, layer_depth=256, is_training=is_training)
    with tf.name_scope("conv2_1_d"):
        W_conv2_1_d = tf.Variable(tf.truncated_normal([1, 1, 64, 256], stddev=0.1), trainable=True, name="W_conv2_1_d")
        b_conv2_1_d = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv2_1_d")
        h_conv2_1_d = tf.nn.bias_add(tf.nn.conv2d(h_conv1_mp, W_conv2_1_d, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_1_d)
    with tf.name_scope("conv2_1"):
        h_conv2_1 = tf.nn.relu(tf.add(h_conv2_1_c_bn, h_conv2_1_d))
        
    with tf.name_scope("conv2_2_a"):
        W_conv2_2_a = tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1), trainable=True, name="W_conv2_2_a")
        b_conv2_2_a = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_2_a")
        h_conv2_2_a = tf.nn.bias_add(tf.nn.conv2d(h_conv2_1, W_conv2_2_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_2_a)
        h_conv2_2_a_bn = conv_batch_normalization(prev_layer=h_conv2_2_a, layer_depth=64, is_training=is_training)
        h_conv2_2_a_bn_relu = tf.nn.relu(h_conv2_2_a_bn)
    with tf.name_scope("conv2_2_b"):
        W_conv2_2_b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), trainable=True, name="W_conv2_2_b")
        b_conv2_2_b = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_2_b")
        h_conv2_2_b = tf.nn.bias_add(tf.nn.conv2d(h_conv2_2_a_bn_relu, W_conv2_2_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_2_b)
        h_conv2_2_b_bn = conv_batch_normalization(prev_layer=h_conv2_2_b, layer_depth=64, is_training=is_training)
        h_conv2_2_b_bn_relu = tf.nn.relu(h_conv2_2_b_bn)
    with tf.name_scope("conv2_2_c"):
        W_conv2_2_c = tf.Variable(tf.truncated_normal([1, 1, 64, 256], stddev=0.1), trainable=True, name="W_conv2_2_c")
        b_conv2_2_c = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv2_2_c")
        h_conv2_2_c = tf.nn.bias_add(tf.nn.conv2d(h_conv2_2_b_bn_relu, W_conv2_2_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_2_c)
        h_conv2_2_c_bn = conv_batch_normalization(prev_layer=h_conv2_2_c, layer_depth=256, is_training=is_training)
    with tf.name_scope("conv2_2"):
        h_conv2_2 = tf.nn.relu(tf.add(h_conv2_2_c_bn, h_conv2_1))
        
    with tf.name_scope("conv2_3_a"):
        W_conv2_3_a = tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1), trainable=True, name="W_conv2_3_a")
        b_conv2_3_a = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_3_a")
        h_conv2_3_a = tf.nn.bias_add(tf.nn.conv2d(h_conv2_2, W_conv2_3_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_3_a)
        h_conv2_3_a_bn = conv_batch_normalization(prev_layer=h_conv2_3_a, layer_depth=64, is_training=is_training)
        h_conv2_3_a_bn_relu = tf.nn.relu(h_conv2_3_a_bn)
    with tf.name_scope("conv2_3_b"):
        W_conv2_3_b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), trainable=True, name="W_conv2_3_b")
        b_conv2_3_b = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2_3_b")
        h_conv2_3_b = tf.nn.bias_add(tf.nn.conv2d(h_conv2_3_a_bn_relu, W_conv2_3_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_3_b)
        h_conv2_3_b_bn = conv_batch_normalization(prev_layer=h_conv2_3_b, layer_depth=64, is_training=is_training)
        h_conv2_3_b_bn_relu = tf.nn.relu(h_conv2_3_b_bn)
    with tf.name_scope("conv2_3_c"):
        W_conv2_3_c = tf.Variable(tf.truncated_normal([1, 1, 64, 256], stddev=0.1), trainable=True, name="W_conv2_3_c")
        b_conv2_3_c = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv2_3_c")
        h_conv2_3_c = tf.nn.bias_add(tf.nn.conv2d(h_conv2_3_b_bn_relu, W_conv2_3_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv2_3_c)
        h_conv2_3_c_bn = conv_batch_normalization(prev_layer=h_conv2_3_c, layer_depth=256, is_training=is_training)
    with tf.name_scope("conv2_3"):
        h_conv2_3 = tf.nn.relu(tf.add(h_conv2_3_c_bn, h_conv2_2))
        
    with tf.name_scope("conv3_1_a"):
        W_conv3_1_a = tf.Variable(tf.truncated_normal([1, 1, 256, 128], stddev=0.1), trainable=True, name="W_conv3_1_a")
        b_conv3_1_a = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_1_a")
        h_conv3_1_a = tf.nn.bias_add(tf.nn.conv2d(h_conv2_3, W_conv3_1_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_1_a)
        h_conv3_1_a_bn = conv_batch_normalization(prev_layer=h_conv3_1_a, layer_depth=128, is_training=is_training)
        h_conv3_1_a_bn_relu = tf.nn.relu(h_conv3_1_a_bn)
    with tf.name_scope("conv3_1_b"):
        W_conv3_1_b = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), trainable=True, name="W_conv3_1_b")
        b_conv3_1_b = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_1_b")
        h_conv3_1_b = tf.nn.bias_add(tf.nn.conv2d(h_conv3_1_a_bn_relu, W_conv3_1_b, strides=[1, 2, 2, 1], padding="SAME"), b_conv3_1_b)
        h_conv3_1_b_bn = conv_batch_normalization(prev_layer=h_conv3_1_b, layer_depth=128, is_training=is_training)
        h_conv3_1_b_bn_relu = tf.nn.relu(h_conv3_1_b_bn)
    with tf.name_scope("conv3_1_c"):
        W_conv3_1_c = tf.Variable(tf.truncated_normal([1, 1, 128, 512], stddev=0.1), trainable=True, name="W_conv3_1_c")
        b_conv3_1_c = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv3_1_c")
        h_conv3_1_c = tf.nn.bias_add(tf.nn.conv2d(h_conv3_1_b_bn_relu, W_conv3_1_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_1_c)
        h_conv3_1_c_bn = conv_batch_normalization(prev_layer=h_conv3_1_c, layer_depth=512, is_training=is_training)
    with tf.name_scope("conv3_1_d"):
        W_conv3_1_d = tf.Variable(tf.truncated_normal([1, 1, 256, 512], stddev=0.1), trainable=True, name="W_conv3_1_d")
        b_conv3_1_d = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv3_1_d")
        h_conv3_1_d = tf.nn.bias_add(tf.nn.conv2d(h_conv2_3, W_conv3_1_d, strides=[1, 2, 2, 1], padding="SAME"), b_conv3_1_d)
    with tf.name_scope("conv3_1"):
        h_conv3_1 = tf.nn.relu(tf.add(h_conv3_1_c_bn, h_conv3_1_d))

    with tf.name_scope("conv3_2_a"):
        W_conv3_2_a = tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1), trainable=True, name="W_conv3_2_a")
        b_conv3_2_a = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_2_a")
        h_conv3_2_a = tf.nn.bias_add(tf.nn.conv2d(h_conv3_1, W_conv3_2_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_2_a)
        h_conv3_2_a_bn = conv_batch_normalization(prev_layer=h_conv3_2_a, layer_depth=128, is_training=is_training)
        h_conv3_2_a_bn_relu = tf.nn.relu(h_conv3_2_a_bn)
    with tf.name_scope("conv3_2_b"):
        W_conv3_2_b = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), trainable=True, name="W_conv3_2_b")
        b_conv3_2_b = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_2_b")
        h_conv3_2_b = tf.nn.bias_add(tf.nn.conv2d(h_conv3_2_a_bn_relu, W_conv3_2_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_2_b)
        h_conv3_2_b_bn = conv_batch_normalization(prev_layer=h_conv3_2_b, layer_depth=128, is_training=is_training)
        h_conv3_2_b_bn_relu = tf.nn.relu(h_conv3_2_b_bn)
    with tf.name_scope("conv3_2_c"):
        W_conv3_2_c = tf.Variable(tf.truncated_normal([1, 1, 128, 512], stddev=0.1), trainable=True, name="W_conv3_2_c")
        b_conv3_2_c = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv3_2_c")
        h_conv3_2_c = tf.nn.bias_add(tf.nn.conv2d(h_conv3_2_b_bn_relu, W_conv3_2_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_2_c)
        h_conv3_2_c_bn = conv_batch_normalization(prev_layer=h_conv3_2_c, layer_depth=512, is_training=is_training)
    with tf.name_scope("conv3_2"):
        h_conv3_2 = tf.nn.relu(tf.add(h_conv3_2_c_bn, h_conv3_1))
        
    with tf.name_scope("conv3_3_a"):
        W_conv3_3_a = tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1), trainable=True, name="W_conv3_3_a")
        b_conv3_3_a = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_3_a")
        h_conv3_3_a = tf.nn.bias_add(tf.nn.conv2d(h_conv3_2, W_conv3_3_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_3_a)
        h_conv3_3_a_bn = conv_batch_normalization(prev_layer=h_conv3_3_a, layer_depth=128, is_training=is_training)
        h_conv3_3_a_bn_relu = tf.nn.relu(h_conv3_3_a_bn)
    with tf.name_scope("conv3_3_b"):
        W_conv3_3_b = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), trainable=True, name="W_conv3_3_b")
        b_conv3_3_b = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_3_b")
        h_conv3_3_b = tf.nn.bias_add(tf.nn.conv2d(h_conv3_3_a_bn_relu, W_conv3_3_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_3_b)
        h_conv3_3_b_bn = conv_batch_normalization(prev_layer=h_conv3_3_b, layer_depth=128, is_training=is_training)
        h_conv3_3_b_bn_relu = tf.nn.relu(h_conv3_3_b_bn)
    with tf.name_scope("conv3_3_c"):
        W_conv3_3_c = tf.Variable(tf.truncated_normal([1, 1, 128, 512], stddev=0.1), trainable=True, name="W_conv3_3_c")
        b_conv3_3_c = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv3_3_c")
        h_conv3_3_c = tf.nn.bias_add(tf.nn.conv2d(h_conv3_3_b_bn_relu, W_conv3_3_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_3_c)
        h_conv3_3_c_bn = conv_batch_normalization(prev_layer=h_conv3_3_c, layer_depth=512, is_training=is_training)
    with tf.name_scope("conv3_3"):
        h_conv3_3 = tf.nn.relu(tf.add(h_conv3_3_c_bn, h_conv3_2))
        
    with tf.name_scope("conv3_4_a"):
        W_conv3_4_a = tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1), trainable=True, name="W_conv3_4_a")
        b_conv3_4_a = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_4_a")
        h_conv3_4_a = tf.nn.bias_add(tf.nn.conv2d(h_conv3_3, W_conv3_4_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_4_a)
        h_conv3_4_a_bn = conv_batch_normalization(prev_layer=h_conv3_4_a, layer_depth=128, is_training=is_training)
        h_conv3_4_a_bn_relu = tf.nn.relu(h_conv3_4_a_bn)
    with tf.name_scope("conv3_4_b"):
        W_conv3_4_b = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), trainable=True, name="W_conv3_4_b")
        b_conv3_4_b = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3_4_b")
        h_conv3_4_b = tf.nn.bias_add(tf.nn.conv2d(h_conv3_4_a_bn_relu, W_conv3_4_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_4_b)
        h_conv3_4_b_bn = conv_batch_normalization(prev_layer=h_conv3_4_b, layer_depth=128, is_training=is_training)
        h_conv3_4_b_bn_relu = tf.nn.relu(h_conv3_4_b_bn)
    with tf.name_scope("conv3_4_c"):
        W_conv3_4_c = tf.Variable(tf.truncated_normal([1, 1, 128, 512], stddev=0.1), trainable=True, name="W_conv3_4_c")
        b_conv3_4_c = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv3_4_c")
        h_conv3_4_c = tf.nn.bias_add(tf.nn.conv2d(h_conv3_4_b_bn_relu, W_conv3_4_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv3_4_c)
        h_conv3_4_c_bn = conv_batch_normalization(prev_layer=h_conv3_4_c, layer_depth=512, is_training=is_training)
    with tf.name_scope("conv3_4"):
        h_conv3_4 = tf.nn.relu(tf.add(h_conv3_4_c_bn, h_conv3_3))
        
    with tf.name_scope("conv4_1_a"):
        W_conv4_1_a = tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.1), trainable=True, name="W_conv4_1_a")
        b_conv4_1_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_1_a")
        h_conv4_1_a = tf.nn.bias_add(tf.nn.conv2d(h_conv3_4, W_conv4_1_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_1_a)
        h_conv4_1_a_bn = conv_batch_normalization(prev_layer=h_conv4_1_a, layer_depth=256, is_training=is_training)
        h_conv4_1_a_bn_relu = tf.nn.relu(h_conv4_1_a_bn)
    with tf.name_scope("conv4_1_b"):
        W_conv4_1_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_1_b")
        b_conv4_1_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_1_b")
        h_conv4_1_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_1_a_bn_relu, W_conv4_1_b, strides=[1, 2, 2, 1], padding="SAME"), b_conv4_1_b)
        h_conv4_1_b_bn = conv_batch_normalization(prev_layer=h_conv4_1_b, layer_depth=256, is_training=is_training)
        h_conv4_1_b_bn_relu = tf.nn.relu(h_conv4_1_b_bn)
    with tf.name_scope("conv4_1_c"):
        W_conv4_1_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_1_c")
        b_conv4_1_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_1_c")
        h_conv4_1_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_1_b_bn_relu, W_conv4_1_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_1_c)
        h_conv4_1_c_bn = conv_batch_normalization(prev_layer=h_conv4_1_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_1_d"):
        W_conv4_1_d = tf.Variable(tf.truncated_normal([1, 1, 512, 1024], stddev=0.1), trainable=True, name="W_conv4_1_d")
        b_conv4_1_d = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_1_d")
        h_conv4_1_d = tf.nn.bias_add(tf.nn.conv2d(h_conv3_4, W_conv4_1_d, strides=[1, 2, 2, 1], padding="SAME"), b_conv4_1_d)
    with tf.name_scope("conv4_1"):
        h_conv4_1 = tf.nn.relu(tf.add(h_conv4_1_c_bn, h_conv4_1_d))
        
    with tf.name_scope("conv4_2_a"):
        W_conv4_2_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1), trainable=True, name="W_conv4_2_a")
        b_conv4_2_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_2_a")
        h_conv4_2_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_1, W_conv4_2_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_2_a)
        h_conv4_2_a_bn = conv_batch_normalization(prev_layer=h_conv4_2_a, layer_depth=256, is_training=is_training)
        h_conv4_2_a_bn_relu = tf.nn.relu(h_conv4_2_a_bn)
    with tf.name_scope("conv4_2_b"):
        W_conv4_2_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_2_b")
        b_conv4_2_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_2_b")
        h_conv4_2_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_2_a_bn_relu, W_conv4_2_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_2_b)
        h_conv4_2_b_bn = conv_batch_normalization(prev_layer=h_conv4_2_b, layer_depth=256, is_training=is_training)
        h_conv4_2_b_bn_relu = tf.nn.relu(h_conv4_2_b_bn)
    with tf.name_scope("conv4_2_c"):
        W_conv4_2_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_2_c")
        b_conv4_2_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_2_c")
        h_conv4_2_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_2_b_bn_relu, W_conv4_2_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_2_c)
        h_conv4_2_c_bn = conv_batch_normalization(prev_layer=h_conv4_2_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_2"):
        h_conv4_2 = tf.nn.relu(tf.add(h_conv4_2_c_bn, h_conv4_1))
        
    with tf.name_scope("conv4_3_a"):
        W_conv4_3_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1), trainable=True, name="W_conv4_3_a")
        b_conv4_3_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_3_a")
        h_conv4_3_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_2, W_conv4_3_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_3_a)
        h_conv4_3_a_bn = conv_batch_normalization(prev_layer=h_conv4_3_a, layer_depth=256, is_training=is_training)
        h_conv4_3_a_bn_relu = tf.nn.relu(h_conv4_3_a_bn)
    with tf.name_scope("conv4_3_b"):
        W_conv4_3_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_3_b")
        b_conv4_3_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_3_b")
        h_conv4_3_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_3_a_bn_relu, W_conv4_3_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_3_b)
        h_conv4_3_b_bn = conv_batch_normalization(prev_layer=h_conv4_3_b, layer_depth=256, is_training=is_training)
        h_conv4_3_b_bn_relu = tf.nn.relu(h_conv4_3_b_bn)
    with tf.name_scope("conv4_3_c"):
        W_conv4_3_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_3_c")
        b_conv4_3_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_3_c")
        h_conv4_3_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_3_b_bn_relu, W_conv4_3_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_3_c)
        h_conv4_3_c_bn = conv_batch_normalization(prev_layer=h_conv4_3_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_3"):
        h_conv4_3 = tf.nn.relu(tf.add(h_conv4_3_c_bn, h_conv4_2))
        
    with tf.name_scope("conv4_4_a"):
        W_conv4_4_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1), trainable=True, name="W_conv4_4_a")
        b_conv4_4_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_4_a")
        h_conv4_4_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_3, W_conv4_4_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_4_a)
        h_conv4_4_a_bn = conv_batch_normalization(prev_layer=h_conv4_4_a, layer_depth=256, is_training=is_training)
        h_conv4_4_a_bn_relu = tf.nn.relu(h_conv4_4_a_bn)
    with tf.name_scope("conv4_4_b"):
        W_conv4_4_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_4_b")
        b_conv4_4_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_4_b")
        h_conv4_4_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_4_a_bn_relu, W_conv4_4_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_4_b)
        h_conv4_4_b_bn = conv_batch_normalization(prev_layer=h_conv4_4_b, layer_depth=256, is_training=is_training)
        h_conv4_4_b_bn_relu = tf.nn.relu(h_conv4_4_b_bn)
    with tf.name_scope("conv4_4_c"):
        W_conv4_4_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_4_c")
        b_conv4_4_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_4_c")
        h_conv4_4_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_4_b_bn_relu, W_conv4_4_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_4_c)
        h_conv4_4_c_bn = conv_batch_normalization(prev_layer=h_conv4_4_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_4"):
        h_conv4_4 = tf.nn.relu(tf.add(h_conv4_4_c_bn, h_conv4_3))
        
    with tf.name_scope("conv4_5_a"):
        W_conv4_5_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1), trainable=True, name="W_conv4_5_a")
        b_conv4_5_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_5_a")
        h_conv4_5_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_4, W_conv4_5_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_5_a)
        h_conv4_5_a_bn = conv_batch_normalization(prev_layer=h_conv4_5_a, layer_depth=256, is_training=is_training)
        h_conv4_5_a_bn_relu = tf.nn.relu(h_conv4_5_a_bn)
    with tf.name_scope("conv4_5_b"):
        W_conv4_5_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_5_b")
        b_conv4_5_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_5_b")
        h_conv4_5_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_5_a_bn_relu, W_conv4_5_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_5_b)
        h_conv4_5_b_bn = conv_batch_normalization(prev_layer=h_conv4_5_b, layer_depth=256, is_training=is_training)
        h_conv4_5_b_bn_relu = tf.nn.relu(h_conv4_5_b_bn)
    with tf.name_scope("conv4_5_c"):
        W_conv4_5_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_5_c")
        b_conv4_5_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_5_c")
        h_conv4_5_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_5_b_bn_relu, W_conv4_5_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_5_c)
        h_conv4_5_c_bn = conv_batch_normalization(prev_layer=h_conv4_5_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_5"):
        h_conv4_5 = tf.nn.relu(tf.add(h_conv4_5_c_bn, h_conv4_4))
        
    with tf.name_scope("conv4_6_a"):
        W_conv4_6_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 256], stddev=0.1), trainable=True, name="W_conv4_6_a")
        b_conv4_6_a = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_6_a")
        h_conv4_6_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_5, W_conv4_6_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_6_a)
        h_conv4_6_a_bn = conv_batch_normalization(prev_layer=h_conv4_6_a, layer_depth=256, is_training=is_training)
        h_conv4_6_a_bn_relu = tf.nn.relu(h_conv4_6_a_bn)
    with tf.name_scope("conv4_6_b"):
        W_conv4_6_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), trainable=True, name="W_conv4_6_b")
        b_conv4_6_b = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4_6_b")
        h_conv4_6_b = tf.nn.bias_add(tf.nn.conv2d(h_conv4_6_a_bn_relu, W_conv4_6_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_6_b)
        h_conv4_6_b_bn = conv_batch_normalization(prev_layer=h_conv4_6_b, layer_depth=256, is_training=is_training)
        h_conv4_6_b_bn_relu = tf.nn.relu(h_conv4_6_b_bn)
    with tf.name_scope("conv4_6_c"):
        W_conv4_6_c = tf.Variable(tf.truncated_normal([1, 1, 256, 1024], stddev=0.1), trainable=True, name="W_conv4_6_c")
        b_conv4_6_c = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_conv4_6_c")
        h_conv4_6_c = tf.nn.bias_add(tf.nn.conv2d(h_conv4_6_b_bn_relu, W_conv4_6_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv4_6_c)
        h_conv4_6_c_bn = conv_batch_normalization(prev_layer=h_conv4_6_c, layer_depth=1024, is_training=is_training)
    with tf.name_scope("conv4_6"):
        h_conv4_6 = tf.nn.relu(tf.add(h_conv4_6_c_bn, h_conv4_5))
        
    with tf.name_scope("conv5_1_a"):
        W_conv5_1_a = tf.Variable(tf.truncated_normal([1, 1, 1024, 512], stddev=0.1), trainable=True, name="W_conv5_1_a")
        b_conv5_1_a = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_1_a")
        h_conv5_1_a = tf.nn.bias_add(tf.nn.conv2d(h_conv4_6, W_conv5_1_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_1_a)
        h_conv5_1_a_bn = conv_batch_normalization(prev_layer=h_conv5_1_a, layer_depth=512, is_training=is_training)
        h_conv5_1_a_bn_relu = tf.nn.relu(h_conv5_1_a_bn)
    with tf.name_scope("conv5_1_b"):
        W_conv5_1_b = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), trainable=True, name="W_conv5_1_b")
        b_conv5_1_b = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_1_b")
        h_conv5_1_b = tf.nn.bias_add(tf.nn.conv2d(h_conv5_1_a_bn_relu, W_conv5_1_b, strides=[1, 2, 2, 1], padding="SAME"), b_conv5_1_b)
        h_conv5_1_b_bn = conv_batch_normalization(prev_layer=h_conv5_1_b, layer_depth=512, is_training=is_training)
        h_conv5_1_b_bn_relu = tf.nn.relu(h_conv5_1_b_bn)
    with tf.name_scope("conv5_1_c"):
        W_conv5_1_c = tf.Variable(tf.truncated_normal([1, 1, 512, 2048], stddev=0.1), trainable=True, name="W_conv5_1_c")
        b_conv5_1_c = tf.Variable(tf.constant(0.0, shape=[2048]), trainable=True, name="b_conv5_1_c")
        h_conv5_1_c = tf.nn.bias_add(tf.nn.conv2d(h_conv5_1_b_bn_relu, W_conv5_1_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_1_c)
        h_conv5_1_c_bn = conv_batch_normalization(prev_layer=h_conv5_1_c, layer_depth=2048, is_training=is_training)
    with tf.name_scope("conv5_1_d"):
        W_conv5_1_d = tf.Variable(tf.truncated_normal([1, 1, 512, 2048], stddev=0.1), trainable=True, name="W_conv5_1_d")
        b_conv5_1_d = tf.Variable(tf.constant(0.0, shape=[2048]), trainable=True, name="b_conv5_1_d")
        h_conv5_1_d = tf.nn.bias_add(tf.nn.conv2d(h_conv4_6, W_conv5_1_d, strides=[1, 2, 2, 1], padding="SAME"), b_conv5_1_d)
    with tf.name_scope("conv5_1"):
        h_conv5_1 = tf.nn.relu(tf.add(h_conv5_1_c_bn, h_conv5_1_d))
        
    with tf.name_scope("conv5_2_a"):
        W_conv5_2_a = tf.Variable(tf.truncated_normal([1, 1, 2048, 512], stddev=0.1), trainable=True, name="W_conv5_2_a")
        b_conv5_2_a = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_2_a")
        h_conv5_2_a = tf.nn.bias_add(tf.nn.conv2d(h_conv5_1, W_conv5_2_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_2_a)
        h_conv5_2_a_bn = conv_batch_normalization(prev_layer=h_conv5_2_a, layer_depth=512, is_training=is_training)
        h_conv5_2_a_bn_relu = tf.nn.relu(h_conv5_2_a_bn)
    with tf.name_scope("conv5_2_b"):
        W_conv5_2_b = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), trainable=True, name="W_conv5_2_b")
        b_conv5_2_b = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_2_b")
        h_conv5_2_b = tf.nn.bias_add(tf.nn.conv2d(h_conv5_2_a_bn_relu, W_conv5_2_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_2_b)
        h_conv5_2_b_bn = conv_batch_normalization(prev_layer=h_conv5_2_b, layer_depth=512, is_training=is_training)
        h_conv5_2_b_bn_relu = tf.nn.relu(h_conv5_2_b_bn)
    with tf.name_scope("conv5_2_c"):
        W_conv5_2_c = tf.Variable(tf.truncated_normal([1, 1, 512, 2048], stddev=0.1), trainable=True, name="W_conv5_2_c")
        b_conv5_2_c = tf.Variable(tf.constant(0.0, shape=[2048]), trainable=True, name="b_conv5_2_c")
        h_conv5_2_c = tf.nn.bias_add(tf.nn.conv2d(h_conv5_2_b_bn_relu, W_conv5_2_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_2_c)
        h_conv5_2_c_bn = conv_batch_normalization(prev_layer=h_conv5_2_c, layer_depth=2048, is_training=is_training)
    with tf.name_scope("conv5_2"):
        h_conv5_2 = tf.nn.relu(tf.add(h_conv5_2_c_bn, h_conv5_1))
        
    with tf.name_scope("conv5_3_a"):
        W_conv5_3_a = tf.Variable(tf.truncated_normal([1, 1, 2048, 512], stddev=0.1), trainable=True, name="W_conv5_3_a")
        b_conv5_3_a = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_3_a")
        h_conv5_3_a = tf.nn.bias_add(tf.nn.conv2d(h_conv5_2, W_conv5_3_a, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_3_a)
        h_conv5_3_a_bn = conv_batch_normalization(prev_layer=h_conv5_3_a, layer_depth=512, is_training=is_training)
        h_conv5_3_a_bn_relu = tf.nn.relu(h_conv5_3_a_bn)
    with tf.name_scope("conv5_3_b"):
        W_conv5_3_b = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), trainable=True, name="W_conv5_3_b")
        b_conv5_3_b = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True, name="b_conv5_3_b")
        h_conv5_3_b = tf.nn.bias_add(tf.nn.conv2d(h_conv5_3_a_bn_relu, W_conv5_3_b, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_3_b)
        h_conv5_3_b_bn = conv_batch_normalization(prev_layer=h_conv5_3_b, layer_depth=512, is_training=is_training)
        h_conv5_3_b_bn_relu = tf.nn.relu(h_conv5_3_b_bn)
    with tf.name_scope("conv5_3_c"):
        W_conv5_3_c = tf.Variable(tf.truncated_normal([1, 1, 512, 2048], stddev=0.1), trainable=True, name="W_conv5_3_c")
        b_conv5_3_c = tf.Variable(tf.constant(0.0, shape=[2048]), trainable=True, name="b_conv5_3_c")
        h_conv5_3_c = tf.nn.bias_add(tf.nn.conv2d(h_conv5_3_b_bn_relu, W_conv5_3_c, strides=[1, 1, 1, 1], padding="SAME"), b_conv5_3_c)
        h_conv5_3_c_bn = conv_batch_normalization(prev_layer=h_conv5_3_c, layer_depth=2048, is_training=is_training)
    with tf.name_scope("conv5_3"):
        h_conv5_3 = tf.nn.relu(tf.add(h_conv5_3_c_bn, h_conv5_2))

    with tf.name_scope("output"):
        inception_output = tf.nn.avg_pool(h_conv5_3, [1, 7, 7, 1], [1, 1, 1, 1], padding="VALID")
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
    tf.summary.FileWriter("logs/", sess.graph)
    # tf.train.write_graph(sess.graph_def, "", "graph.pbtxt")
    # with tf.gfile.GFile("graph.pb", mode="wb") as f:
    #     f.write(convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"]).SerializeToString())
