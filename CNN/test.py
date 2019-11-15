import tensorflow as tf

batch_size = 5
validation_size = 4686
validation_tfrecord = "flowers_validation.tfrecord"

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features["label"], tf.int64)
    return img, label

with tf.Session() as sess:
    tf.train.import_meta_graph("ckpt/model.ckpt-200.meta")
    dataset_validation = tf.data.TFRecordDataset([validation_tfrecord])
    dataset_validation = dataset_validation.map(read_and_decode)
    dataset_validation = dataset_validation.repeat(1).shuffle(1000).batch(batch_size)
    iterator_validation = dataset_validation.make_initializable_iterator()
    next_element_validation = iterator_validation.get_next()
    sess.run(iterator_validation.initializer)

    x_ = sess.graph.get_tensor_by_name("input/x_input:0")
    y_ = sess.graph.get_tensor_by_name("input/y_input:0")
    accuracy = sess.graph.get_tensor_by_name("accuracy:0")
    acc = 0.0
    for step in range(int(validation_size/batch_size)):
        img_validation, label_validation = sess.run(next_element_validation)
        acc += sess.run(accuracy, feed_dict={x_: img_validation, y_: label_validation})
    print("test accuracy:[%.2f]"%(acc/int(validation_size/batch_size)))