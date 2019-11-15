import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate=0.01
max_samples=40000
batch_size=128
display_step=10

n_input=28
n_steps=28
n_hidden11=100
n_hidden12=120
n_hidden21=200
n_hidden22=220
n_classes=10

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden12+n_hidden22, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def BiRNN(x, weights, biases):
    # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*2, state_is_tuple=True)
    # state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
    # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * 2, state_is_tuple=True)
    # state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
    fw_layers = [tf.nn.rnn_cell.BasicLSTMCell(n_hidden11, forget_bias=1.0), tf.nn.rnn_cell.BasicLSTMCell(n_hidden12, forget_bias=1.0)]
    bw_layers = [tf.nn.rnn_cell.BasicLSTMCell(n_hidden21, forget_bias=1.0), tf.nn.rnn_cell.BasicLSTMCell(n_hidden22, forget_bias=1.0)]

    fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_layers)
    state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
    bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_layers)
    state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)

    # # outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, state_fw, state_bw)
    # # return tf.matmul(outputs[-1], weights) + biases
    # (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, initial_state_fw=state_fw,
    #                                                            initial_state_bw=state_bw)
    # print(outputs, output_states[0][1], output_states[1][1])
    # _inputs = tf.concat([output_states[0][1].h, output_states[1][1].h], 1)
    # return tf.matmul(_inputs, weights) + biases
    (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, initial_state_fw=state_fw,
                                                               initial_state_bw=state_bw)
    print(outputs, output_states)
    _inputs = tf.concat([output_states[0][1].h, output_states[1][1].h], 1)
    return tf.matmul(_inputs, weights) + biases


pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

corrected_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter ' + str(step * batch_size) + ',Minibatch Loss=' + '{:.6f}'.format(
                loss) + ',Training Accuracy= ' + '{:.5f}'.format(acc))
        step += 1
    print('Optimizatin Finished!')
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print('Testing Accuracy:', sess.run(accuracy, feed_dict={x:test_data, y: test_label}))

### 多层birnn会遇到n_hidden不可以随意更改的问题
### https://blog.csdn.net/qq_27009517/article/details/82345134
### https://stackoverflow.com/questions/48792485/value-error-from-tf-nn-dynamic-rnn-dimensions-must-be-equal