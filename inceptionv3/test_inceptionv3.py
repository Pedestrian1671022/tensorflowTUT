import os
import tensorflow as tf
# from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_pixels = 299
batch_size = 1
validation_size = 1668
validation_tfrecord = "flowers_validation.tfrecord"


def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    label = tf.cast(features["label"], tf.int64)

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, label, filename

with tf.Session() as sess:
    tf.train.import_meta_graph("ckpt/model.ckpt.meta")
    tf.train.Saver().restore(sess, "ckpt/model.ckpt")
    dataset_validation = tf.data.TFRecordDataset([validation_tfrecord])
    dataset_validation = dataset_validation.map(read_and_decode)
    dataset_validation = dataset_validation.repeat(1).shuffle(1000).batch(batch_size)
    iterator_validation = dataset_validation.make_initializable_iterator()
    next_element_validation = iterator_validation.get_next()
    sess.run(iterator_validation.initializer)

    x_ = sess.graph.get_tensor_by_name("input/x_input:0")
    y_ = sess.graph.get_tensor_by_name("input/y_input:0")
    rate = sess.graph.get_tensor_by_name("InceptionV3/output/rate:0")
    is_training = sess.graph.get_tensor_by_name("input/is_training:0")
    accuracy = sess.graph.get_tensor_by_name("accuracy:0")
    acc = 0.0
    for step in range(int(validation_size/batch_size)):
        img_validation, label_validation, filename = sess.run(next_element_validation)
        acc += sess.run(accuracy, feed_dict={x_: img_validation, y_: label_validation, is_training:False, rate:0})
    print("test accuracy:[%.4f]"%(acc/int(validation_size/batch_size)))
    tf.summary.FileWriter("logs/", sess.graph)
    # with tf.gfile.GFile("graph.pb", mode="wb") as f:
    #     f.write(convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"]).SerializeToString())