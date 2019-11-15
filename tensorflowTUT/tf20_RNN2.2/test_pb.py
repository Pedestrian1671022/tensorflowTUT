import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
classes = 10
n_steps = 28
n_inputs = 28
training_iters = 100000

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
    with tf.gfile.GFile("graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

    tf.global_variables_initializer().run()
    x_ = sess.graph.get_tensor_by_name("import/x_input:0")
    y_ = sess.graph.get_tensor_by_name("import/y_input:0")
    accuracy = sess.graph.get_tensor_by_name("import/accuracy:0")

    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        if step % 20 ==0:
            print (sess.run(accuracy, feed_dict={
            x_: batch_xs,
            y_: batch_ys
        }))