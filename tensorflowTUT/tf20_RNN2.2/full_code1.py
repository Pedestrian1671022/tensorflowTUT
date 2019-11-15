import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28  # shape 28*28
n_steps = 28  # time steps
n_hidden_unis = 100  # neurons in hidden layer
n_classes = 10  # classes 0-9

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="x_input")
y = tf.placeholder(tf.float32, [None, n_classes], name="y_input")

# Define weights
weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, 120])),
    # (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[120, ])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):

    # hidden layer for input to cell
    # X(128 batch, 28 steps, 28 inputs) => (128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    # ==>(128 batch * 28 steps, 28 hidden)
    X_in = tf.matmul(X, weights['in'])+biases['in']
    # ==>(128 batch , 28 steps, 128 hidden)
    X_in = tf.reshape(X_in,[-1, n_steps, 120])
    # X_in = tf.transpose(X_in, [1,0,2])


    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    print(_init_state)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=True)
    # outputs, states = tf.nn .dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    print(outputs, states.c, states.h)


    # hidden layer for output as the final results
    results = tf.matmul(states.h, weights['out']) + biases['out']  # states[1]->m_state states[1]=output[-1]
    # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # return results, outputs, states.c, states.h
    return results


# pred, outputs, cstates, hstates = RNN(x, weights, biases)
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

saver = tf.train.Saver(tf.global_variables())
saver=tf.train.Saver(max_to_keep=3)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        # print(sess.run([train_op, outputs, cstates, hstates], feed_dict={
        #     x: batch_xs,
        #     y: batch_ys
        # }))
        # print(sess.run(outputs, cstates, hstates))
        saver.save(sess, './ckpt/mnist.ckpt', global_step=step+1)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        if step % 20 ==0:
            print (sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys
        }))
        step += 1
    tf.summary.FileWriter("logs/", sess.graph)
    tf.train.write_graph(sess.graph_def, "", "graph.pbtxt")
    with tf.gfile.GFile("graph.pb", mode="wb") as f:
        f.write(convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"]).SerializeToString())
    # saver.restore(sess, tf.train.latest_checkpoint(sess,
    #                                                '/home/pedestrian/Documents/tutorials/tensorflowTUT/tf20_RNN2.2/ckpt/checkpoint'))
    # # saver = tf.train.import_meta_graph('/home/pedestrian/Documents/tutorials/tensorflowTUT/tf20_RNN2.2/ckpt/mnist.ckpt-780.meta')
    # # saver.restore(sess, tf.train.latest_checkpoint('/home/pedestrian/Documents/tutorials/tensorflowTUT/tf20_RNN2.2/ckpt'))
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    # print (sess.run(accuracy, feed_dict={
    # x: batch_xs,
    # y: batch_ys
    # }))