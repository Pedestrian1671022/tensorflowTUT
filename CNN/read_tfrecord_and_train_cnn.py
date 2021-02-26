import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

image_pixels = 224
batch_size = 50
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

with tf.name_scope("input"):
    x_ = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="x_input")
    y_ = tf.placeholder(tf.int64, [None], name="y_input")

with tf.name_scope("conv"):
    with tf.name_scope("conv1"):
        W_con1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1), trainable=True, name="W_con1")
        b_con1 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True, name="b_con1")
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_, W_con1, strides=[1, 1, 1, 1], padding="SAME") + b_con1, "h_conv1")
        h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="h_pool1")
    with tf.name_scope("conv2"):
        W_con2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), trainable=True, name="W_conv2")
        b_con2 = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b_conv2")
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_con2, strides=[1, 1, 1, 1], padding="SAME") + b_con2, name="h_conv2")
        h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="h_pool2")
    with tf.name_scope("conv3"):
        W_con3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), trainable=True, name="W_conv3")
        b_con3 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b_conv3")
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_con3, strides=[1, 1, 1, 1], padding="SAME") + b_con3, name="h_conv3")
        h_pool3 = tf.nn.max_pool(h_conv3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="h_pool3")
    with tf.name_scope("conv4"):
        W_con4 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), trainable=True, name="W_conv4")
        b_con4 = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True, name="b_conv4")
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_con4, strides=[1, 1, 1, 1], padding="SAME") + b_con4, name="h_conv4")
        h_pool4 = tf.nn.max_pool(h_conv4, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="h_pool4")

with tf.name_scope("full_connection_and_output"):
    with tf.name_scope("full_connection"):
        W_fc1 = tf.Variable(tf.truncated_normal([14*14*256, 1024], stddev=0.1), trainable=True, name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.0, shape=[1024]), trainable=True, name="b_fc1")
        h_pool4_flat = tf.reshape(h_pool4, [-1, 14*14*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1, name="h_fc1")
        # rate = tf.placeholder(tf.float32, name="rate")
        # h_fc1_drop = tf.nn.dropout(h_fc1, rate=rate)
    with tf.name_scope("output"):
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 5], stddev=0.1), trainable=True, name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.0, shape=[5]), trainable=True, name="b_fc2")
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

with tf.name_scope("loss"):
    one_hot_labels = slim.one_hot_encoding(y_, classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=one_hot_labels))

with tf.name_scope("train"):
    lr = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss)

update_learning_rate = tf.assign(lr, lr * 0.8)

correct_prediction = tf.equal(y_, tf.argmax(y_conv, 1), name="correct_prediction")
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
            _, _loss, _accuracy = sess.run([train_step, loss, accuracy], feed_dict={x_: img_train, y_: label_train})
            if step % 10 == 0:
                print("step:", step / 10, " loss:", _loss, " accuracy:", _accuracy)
        tf.train.Saver().save(sess, "ckpt/model.ckpt")
        print("save ckpt:", epoch)
    # saver.export_meta_graph("ckpt/model.meta")
    # tf.summary.FileWriter("logs/", sess.graph)
    # tf.train.write_graph(sess.graph_def, "", "graph.pbtxt")
    # with tf.gfile.GFile("graph.pb", mode="wb") as f:
    #     f.write(convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"]).SerializeToString())
