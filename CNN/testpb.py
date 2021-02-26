import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_pixels = 224
batch_size = 50
classes = 5
test_data = 4686
test_steps = int(test_data/batch_size)

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"label":tf.FixedLenFeature([], tf.int64), "image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    label = tf.cast(features["label"], tf.int64)

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, label, filename

with tf.Session() as sess:
    with tf.gfile.GFile("graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

    tf.global_variables_initializer().run()
    x_ = sess.graph.get_tensor_by_name("import/input/x_input:0")
    y_ = sess.graph.get_tensor_by_name("import/input/y_input:0")
    # rate1 = sess.graph.get_tensor_by_name("import/full_connection_and_output/full_connection1/rate1:0")
    # rate2 = sess.graph.get_tensor_by_name("import/full_connection_and_output/full_connection2/rate2:0")
    accuracy = sess.graph.get_tensor_by_name("import/accuracy:0")
    correct_prediction = sess.graph.get_tensor_by_name("import/correct_prediction:0")

    filenames_test = ["flowers_validation.tfrecord"]
    dataset_test = tf.data.TFRecordDataset(filenames_test)
    dataset_test = dataset_test.map(read_and_decode)
    dataset_test = dataset_test.repeat().shuffle(1000).batch(batch_size)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    sess.run(iterator_test.initializer)

    acc = 0.0
    for step in range(test_steps):
        img, label, filename = sess.run(next_element_test)
        _accuracy, _correct_prediction= sess.run([accuracy, correct_prediction], feed_dict={x_: img, y_: label})
        acc += _accuracy
        print(step, _accuracy, _correct_prediction)
        # acc += sess.run(accuracy, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0})
        # print(step, sess.run(accuracy, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0}), sess.run(ll, feed_dict={x_: img, y_: label, rate1: 0.0, rate2: 0.0}))
    print("test accuracy:[%.5f]" % (acc / test_steps))