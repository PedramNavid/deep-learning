import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np


def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        lstm_size: size of the hidden layers
        num_layers: number of LSTM layers
        batch_size: batch size
        keep_prob: scalar tensor for dropout layer

    '''

    def build_cell(lstm_size, keep_prob):
        lstm = rnn.BasicLSTMCell(lstm_size)
        drop = rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    # Stack layers
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits
    '''

    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))


    logits = tf.matmul(x, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimzier(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

# Build the network ----

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Build inputs
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cells
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # Run the data through the RNN layers
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # Get prdictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss and optimizer
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimzier(self.loss, learning_rate, grad_clip)