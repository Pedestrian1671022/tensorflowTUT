import tensorflow as tf


batch_size = 10
classes = 5
test_data = 500
test_steps = int(test_data/batch_size)

def one_hot(labels, label_classes):
    one_hot_labels = [[int(i == int(labels[j])) for i in range(label_classes)] for j in range(len(labels))]
    return one_hot_labels

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features["label"], tf.int32)
    return img, label

with tf.Session() as sess:
    with tf.gfile.GFile("graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

    tf.global_variables_initializer().run()
    x_ = sess.graph.get_tensor_by_name("import/input/x_input:0")
    y_ = sess.graph.get_tensor_by_name("import/input/y_input:0")
    rate1 = sess.graph.get_tensor_by_name("import/full_connection_and_output/full_connection1/rate1:0")
    rate2 = sess.graph.get_tensor_by_name("import/full_connection_and_output/full_connection2/rate2:0")
    y_conv = sess.graph.get_tensor_by_name("import/full_connection_and_output/output/Softmax:0")
    accuracy = sess.graph.get_tensor_by_name("import/accuracy:0")
    ll = sess.graph.get_tensor_by_name("import/correct_prediction:0")

    filenames_test = ["flowers_test.tfrecord"]
    dataset_test = tf.data.TFRecordDataset(filenames_test)
    dataset_test = dataset_test.map(read_and_decode)
    dataset_test = dataset_test.repeat().shuffle(1000).batch(batch_size)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    sess.run(iterator_test.initializer)

    acc = 0.0
    for step in range(test_steps):
        img, label = sess.run(next_element_test)
        label = one_hot(label, classes)
        acc += sess.run(accuracy, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0})
        print(step, sess.run(accuracy, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0}), sess.run(ll, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0}))
    print("test accuracy:[%.5f]" % (acc / test_steps))