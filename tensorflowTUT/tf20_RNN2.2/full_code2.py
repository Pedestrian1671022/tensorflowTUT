import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28  # shape 28*28
n_steps = 28  # time steps
n_hidden_unis1 = 120  # neurons in hidden layer
n_hidden_unis2 = 100  # neurons in hidden layer
n_classes = 10  # classes 0-9

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, 120])),
    # (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis2, n_classes]))
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
    X_in = tf.reshape(X_in, [-1, n_steps, 120])
    print(X_in)
    # X_in = tf.transpose(X_in, [1,0,2])


    # cell
    lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis1, forget_bias=1.0, state_is_tuple=True)
    lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, input_keep_prob=1.0, state_keep_prob=1.0, output_keep_prob=0.8)
    lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis2, forget_bias=1.0, state_is_tuple=True)
    lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, input_keep_prob=1.0, state_keep_prob=1.0,
                                               output_keep_prob=0.8)
    # lstm_layers = [tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0) for _ in range(2)]
    # cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2], state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, h_state)
    _init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, X_in, initial_state=_init_state, time_major=False)
    print(_init_state)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=True)
    # outputs, states = tf.nn .dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    print(outputs, states)


    # hidden layer for output as the final results
    results = tf.matmul(states[1].h, weights['out']) + biases['out']  # states[1]->m_state states[1]=output[-1]
    # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
        saver.save(sess, './ckpt/mnist.ckpt', global_step=step+1)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        if step % 20 ==0:
            print (sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys
        }))
        step += 1
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

### 多层birnn会遇到n_hidden不可以随意更改的问题
### https://blog.csdn.net/qq_27009517/article/details/82345134
### https://stackoverflow.com/questions/48792485/value-error-from-tf-nn-dynamic-rnn-dimensions-must-be-equal