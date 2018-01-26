from model import CharRNN
import tensorflow as tf
import numpy as np
import time

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 20
print_every_n = 100
save_every_n = 200

# Read text
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read()

vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

print("Read %s length text with vocabulary of: %s" % (len(text), len(vocab)))


# Get Batches
def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:n_batches * chars_per_batch]

    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y_temp = arr[:, n + 1:n + n_steps + 1]
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :y_temp.shape[1]] = y_temp

        yield x, y

# Train Model

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state,  _ = sess.run([model.loss,
                                                  model.final_state,
                                                  model.optimizer],
                                                 feed_dict=feed)

            if (counter % print_every_n == 0):
                end = time.time()
                print('Epoch: {}/{}... '.format(e + 1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))